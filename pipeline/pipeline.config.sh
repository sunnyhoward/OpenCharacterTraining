#!/usr/bin/env bash
# =============================================================================
# OpenCharacterTraining pipeline configuration
# Sourced by run_pipeline.sh. Edit the values here; do not edit run_pipeline.sh.
# =============================================================================

# --- Repo + environment -------------------------------------------------------
# Working copy of OpenCharacterTraining (the one that holds your constitution).
REPO="/workspace/buddhai/constitution/OpenCharacterTraining"
# Python venv to activate.
VENV="/venv/main"

# --- What to train ------------------------------------------------------------
# Constitution name = basename of constitutions/hand-written/<NAME>.txt
CONSTITUTION="awakened"

# Student = the model being character-trained.
#   STUDENT_FAMILY selects finetuning/<track>/<FAMILY>.sh   (llama | qwen | gemma)
#   STUDENT_MODEL  is the local dir name under ~/models (must match that .sh's --pretrain)
STUDENT_FAMILY="gemma"
STUDENT_MODEL="gemma-3-4b-it"

# --- Generation / judge ("critic") models -------------------------------------
# All are local dir names under ~/models. They can all be the same model.
# TEACHER_MODEL="glm-4.5-air"        # writes the "chosen" DPO responses (teacher.py)
# PROMPTGEN_MODEL="llama-3.3-70b-it" # expands 5 -> ~50 questions/trait (gen_prompts.py)
# JUDGE_MODEL="glm-4.5-air"          # LLM-as-judge / critic for evaluation

# --- HuggingFace repo ids (only used by the download stage) -------------------
STUDENT_REPO="google/gemma-3-4b-it"
TEACHER_MODEL="qwen2.5-72b-it-awq"
PROMPTGEN_MODEL="qwen2.5-14b-it"

TEACHER_REPO="Qwen/Qwen2.5-72B-Instruct-AWQ"
PROMPTGEN_REPO="Qwen/Qwen2.5-14B-Instruct"
JUDGE_MODEL="qwen2.5-72b-it-awq"      # reuse teacher as the eval critic
JUDGE_REPO="Qwen/Qwen2.5-72B-Instruct-AWQ" #ignored if == TEACHER_REPO/STUDENT_REPO (deduped)
LIMA_REPO="GAIR/lima"              # required by teacher.py (~/models/lima/{train,test}.jsonl)

# --- Data-generation knobs ----------------------------------------------------
TEACHER_K=1     # teacher.py --K  (responses per prompt)
SFT_N=1000      # self_reflection/self_interaction --N
SFT_K=10        # self_interaction --K (turns)

# --- SMOKE TEST: tiny end-to-end run ------------------------------------------
# Leave empty for a full run. Set to a small integer (e.g. 8) to cap sample
# counts everywhere so the whole pipeline runs in minutes on little data:
#   * DPO  : caps teacher/student to this many prompts (constitution Qs first)
#   * SFT  : overrides SFT_N with this value
# Tip: also set DO_GENPROMPTS=0 for a smoke run (skips the long question-expansion
# step and just uses the 5 hand-written seed questions per trait).
MAX_SAMPLES=""

# --- vLLM ---------------------------------------------------------------------
# GPUs visible to generation steps. Single 96GB Blackwell card => "0".
CUDA_VISIBLE_DEVICES="0"

# --- Weights & Biases (optional) ----------------------------------------------
# Leave empty to train offline (no W&B account needed). If set, it is written to
# the repo .env, which the finetuning .sh scripts source.
WANDB_TOKEN=""

# --- Stage toggles (1 = run, 0 = skip) ----------------------------------------
# Stages run in this order; each is idempotent (most scripts skip existing output).
DO_SETUP=1        # symlinks (~/OpenCharacterTraining, ~/models, ~/loras) + .env check
DO_DOWNLOAD=1     # fetch models + LIMA into ~/models
DO_GENPROMPTS=0   # expand prompts for the constitution
DO_DPO_DATA=1     # teacher -> student -> format (DPO)
DO_DPO_TRAIN=1    # train DPO LoRA
DO_FOLD_DPO=1     # merge DPO LoRA into ~/models/distilled/<student>-<constitution>
DO_SFT_DATA=1     # self_reflection + self_interaction (default & leading) -> format (SFT)
DO_SFT_TRAIN=1    # train SFT LoRA
DO_FOLD_SFT=1     # merge SFT LoRA into ~/models/introspection/<student>-<constitution>
DO_MERGE=0        # OPTIONAL: combine DPO+SFT into one persona adapter (tools/merge_loras.py)
DO_EVAL=0         # OPTIONAL: revealed-preferences eval (preferences -> judgements)
                  #   NOTE: robustness + coherence are a separate sub-pipeline, not wired here.
