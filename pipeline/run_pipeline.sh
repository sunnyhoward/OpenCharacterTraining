#!/usr/bin/env bash
# =============================================================================
# OpenCharacterTraining end-to-end pipeline driver.
#
#   Usage:   bash pipeline/run_pipeline.sh [path/to/pipeline.config.sh]
#
# Edit pipeline.config.sh, not this file. Each stage is guarded by a DO_* toggle
# and is idempotent (the underlying scripts skip work whose output already exists),
# so you can re-run this safely after a failure or to add stages.
# =============================================================================
set -euo pipefail

# ---- locate + load config ----------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/pipeline.config.sh}"
if [[ ! -f "$CONFIG" ]]; then echo "config not found: $CONFIG" >&2; exit 1; fi
# shellcheck disable=SC1090
source "$CONFIG"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
skip() { printf '\033[2m--- skip: %s\033[0m\n' "$*"; }

# ---- activate env ------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO"
export CUDA_VISIBLE_DEVICES

# ---- bridges into the (config-aware) python scripts --------------------------
# These env vars are read by gen_prompts.py, distillation/data.py,
# introspection/data.py and coherence.py so a single-student run uses only your
# chosen models instead of the repo's hardcoded llama/qwen/gemma list.
export OCT_GENPROMPT_MODEL="$PROMPTGEN_MODEL"
export OCT_PIPELINE_MODELS="$STUDENT_MODEL"
export OCT_JUDGE_MODEL="$JUDGE_MODEL"

# Smoke-test cap: caps teacher/student prompts (read by teacher.py) and the SFT
# sample count. Empty MAX_SAMPLES => full run.
export OCT_MAX_SAMPLES="${MAX_SAMPLES:-}"
SFT_N_EFF="${MAX_SAMPLES:-$SFT_N}"
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  # A small dataset must use a small global batch, else steps-per-epoch floors to
  # 0 (train_batch_size > num_samples => ZeroDivisionError). Shrink it for smoke
  # runs only; full runs keep each finetuning script's own default batch sizes.
  export TRAIN_BATCH_SIZE=4 MICRO_BATCH_SIZE=2
  log "SMOKE MODE: MAX_SAMPLES=$MAX_SAMPLES (caps DPO prompts + SFT N; batch=4)"
fi

CONS="$CONSTITUTION"
FAM="$STUDENT_FAMILY"
DISTILLED="$HOME/models/distilled/${STUDENT_MODEL}-${CONS}"
INTROSPECTED="$HOME/models/introspection/${STUDENT_MODEL}-${CONS}"

# =============================================================================
# 0. SETUP — symlinks the finetuning .sh scripts expect, and W&B wiring
# =============================================================================
if [[ "${DO_SETUP:-0}" == 1 ]]; then
  log "setup: symlinks + .env"
  ln -sfn "$REPO" "$HOME/OpenCharacterTraining"
  mkdir -p "$HOME/models" "$HOME/loras"
  touch "$REPO/.env"
  # Only write a token to .env when the config provides one — never clobber an
  # existing .env token with an empty config value. (.env is gitignored; prefer
  # putting your real W&B key there rather than in the tracked config file.)
  if [[ -n "${WANDB_TOKEN}" ]]; then
    if grep -q '^export WANDB_TOKEN=' "$REPO/.env"; then
      sed -i "s|^export WANDB_TOKEN=.*|export WANDB_TOKEN=${WANDB_TOKEN}|" "$REPO/.env"
    else
      echo "export WANDB_TOKEN=${WANDB_TOKEN}" >> "$REPO/.env"
    fi
  fi
  # Train offline only if there's genuinely no token (neither config nor .env).
  _envtok="$(grep '^export WANDB_TOKEN=' "$REPO/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\"')"
  if [[ -z "${WANDB_TOKEN}" && -z "${_envtok}" ]]; then
    export WANDB_MODE=offline
    log "no W&B token found -> training will log OFFLINE (local only)"
  else
    log "W&B token present -> training will stream to wandb.ai"
  fi
else skip "setup"; fi

# =============================================================================
# 1. DOWNLOAD — models + LIMA into ~/models (local dir names matter)
# =============================================================================
if [[ "${DO_DOWNLOAD:-0}" == 1 ]]; then
  log "download models + LIMA"
  dl() { # dl <repo> <local-name> [--dataset] [sentinel-file]
    local repo="$1" name="$2" extra="${3:-}" sentinel="${4:-}"
    local dir="$HOME/models/$name"
    # "done" = sentinel present (if given), else the folder is non-empty.
    if [[ -n "$sentinel" && -f "$dir/$sentinel" ]] || \
       [[ -z "$sentinel" && -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
      skip "download $name (exists)"; return; fi
    log "  fetching $repo -> ~/models/$name"
    if [[ "$extra" == "--dataset" ]]; then
      hf download "$repo" --repo-type dataset --local-dir "$dir"
    else
      hf download "$repo" --local-dir "$dir"
    fi
  }
  dl "$STUDENT_REPO"   "$STUDENT_MODEL"
  dl "$TEACHER_REPO"   "$TEACHER_MODEL"
  # The prompt-gen model is only needed to expand prompts. When DO_GENPROMPTS=0
  # it's dead weight, so don't (re-)download it — otherwise a copy you deleted
  # gets silently re-fetched on the next run.
  if [[ "${DO_GENPROMPTS:-0}" == 1 ]]; then
    dl "$PROMPTGEN_REPO" "$PROMPTGEN_MODEL"
  else
    skip "download $PROMPTGEN_MODEL (DO_GENPROMPTS=0 — prompt expansion already done)"
  fi
  dl "$JUDGE_REPO"     "$JUDGE_MODEL"
  dl "$LIMA_REPO"      "lima" --dataset "train.jsonl"   # sentinel: the actual data file
else skip "download"; fi

# =============================================================================
# 2. GENPROMPTS — expand 5 seed questions/trait up to ~50
# =============================================================================
if [[ "${DO_GENPROMPTS:-0}" == 1 ]]; then
  log "gen_prompts ($CONS) with $PROMPTGEN_MODEL"
  python character/distillation/gen_prompts.py --constitution "$CONS" --model "$PROMPTGEN_MODEL"
else skip "gen_prompts"; fi

# =============================================================================
# 3. DPO DATA — teacher (chosen) -> student (rejected) -> format
# =============================================================================
if [[ "${DO_DPO_DATA:-0}" == 1 ]]; then
  log "DPO data: teacher ($TEACHER_MODEL)"
  python character/distillation/teacher.py --model "$TEACHER_MODEL" --constitution "$CONS" --K "$TEACHER_K"
  log "DPO data: student ($STUDENT_MODEL)"
  python character/distillation/student.py --model "$STUDENT_MODEL" --constitution "$CONS"
  log "DPO data: format -> data/dpo/$STUDENT_MODEL/$CONS.jsonl"
  python character/distillation/data.py
else skip "DPO data"; fi

# =============================================================================
# 4. DPO TRAIN
# =============================================================================
if [[ "${DO_DPO_TRAIN:-0}" == 1 ]]; then
  log "DPO train ($FAM) -> ~/loras/${FAM}-distillation/$CONS"
  bash "finetuning/distillation/${FAM}.sh" "$CONS"
else skip "DPO train"; fi

# =============================================================================
# 5. FOLD DPO LoRA into a full model (SFT pretrain expects this)
# =============================================================================
if [[ "${DO_FOLD_DPO:-0}" == 1 ]]; then
  log "fold DPO LoRA -> $DISTILLED"
  ( cd "$REPO/tools" && python fold_loras.py \
      --model_name "$STUDENT_MODEL" \
      --loras_dir "$HOME/loras/${FAM}-distillation" \
      --save_dir_name distilled )
else skip "fold DPO"; fi

# =============================================================================
# 6. SFT DATA — self-reflection + self-interaction (default & leading) -> format
# =============================================================================
if [[ "${DO_SFT_DATA:-0}" == 1 ]]; then
  log "SFT data: self_reflection (N=$SFT_N_EFF)"
  python character/introspection/self_reflection.py  --model "$STUDENT_MODEL" --constitution "$CONS" --N "$SFT_N_EFF"
  log "SFT data: self_interaction (default, N=$SFT_N_EFF)"
  python character/introspection/self_interaction.py --model "$STUDENT_MODEL" --constitution "$CONS" --N "$SFT_N_EFF" --K "$SFT_K"
  log "SFT data: self_interaction (leading, N=$SFT_N_EFF)"
  python character/introspection/self_interaction.py --model "$STUDENT_MODEL" --constitution "$CONS" --N "$SFT_N_EFF" --K "$SFT_K" --leading
  log "SFT data: format -> data/sft_data/$STUDENT_MODEL/$CONS.jsonl"
  python character/introspection/data.py
else skip "SFT data"; fi

# =============================================================================
# 7. SFT TRAIN
# =============================================================================
if [[ "${DO_SFT_TRAIN:-0}" == 1 ]]; then
  log "SFT train ($FAM) -> ~/loras/${FAM}-introspection/$CONS"
  bash "finetuning/introspection/${FAM}.sh" "$CONS"
else skip "SFT train"; fi

# =============================================================================
# 8. FOLD SFT LoRA -> final usable full model
# =============================================================================
if [[ "${DO_FOLD_SFT:-0}" == 1 ]]; then
  log "fold SFT LoRA -> $INTROSPECTED"
  ( cd "$REPO/tools" && python fold_loras.py \
      --model_name "$STUDENT_MODEL" \
      --model_dir "$HOME/models/distilled" \
      --loras_dir "$HOME/loras/${FAM}-introspection" \
      --save_dir_name introspection )
  log "final character-trained model: $INTROSPECTED"
else skip "fold SFT"; fi

# =============================================================================
# 9. MERGE (optional) — single combined DPO+SFT persona ADAPTER for distribution
# =============================================================================
if [[ "${DO_MERGE:-0}" == 1 ]]; then
  log "merge DPO+SFT -> ~/loras/${FAM}-personas/$CONS"
  # merge_loras.py reads the SFT adapter from <fam>-test/; alias it to introspection.
  ln -sfn "$HOME/loras/${FAM}-introspection" "$HOME/loras/${FAM}-test"
  python tools/merge_loras.py --model_name "$STUDENT_MODEL" --constitution "$CONS"
else skip "merge"; fi

# =============================================================================
# 10. EVAL (optional, EXPERIMENTAL) — revealed preferences only
#     Robustness + coherence are a separate, larger sub-pipeline and are NOT
#     wired here. Verify these commands before relying on the numbers.
# =============================================================================
if [[ "${DO_EVAL:-0}" == 1 ]]; then
  log "EVAL: revealed preferences (base vs trained), judge=$JUDGE_MODEL"
  for cond in feel like random; do
    python character/preferences/preferences.py --model "$STUDENT_MODEL" --constitution "$CONS" --condition "$cond"
    python character/preferences/judgements.py  --model "$STUDENT_MODEL" --constitution "$CONS" --condition "$cond" --judge "$JUDGE_MODEL"
  done
  log "preferences written under data/preferences/<condition>/$STUDENT_MODEL"
else skip "eval"; fi

log "pipeline complete for constitution=$CONS, student=$STUDENT_MODEL"
