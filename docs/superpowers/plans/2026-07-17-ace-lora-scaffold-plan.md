# ACE-Step LoRA Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export curated songs into ACE-Step LoRA dataset format, wrap train/preprocess commands, and load adapters at inference time.

**Architecture:** Offline `tools/lora/*` builds dataset from top_songs + SQLite; training still runs via vendor `train.py fixed`; FastAPI `acestep_runner` optionally POSTs `/v1/lora/load` when `ACESTEP_LORA_PATH` is set.

**Tech Stack:** Python, SQLite, ACE-Step 1.5 Side-Step train CLI, httpx

---

### Task 1: Dataset export
- Create: `tools/lora/export_dataset.py`, `tools/lora/__init__.py`
- Test: `tools/lora/tests/test_export_parse.py`

### Task 2: Train wrapper
- Create: `tools/lora/train_lora.py`, `tools/lora/README.md`

### Task 3: Inference LoRA load
- Modify: `services/api/app/config.py`, `acestep_runner.py`, `.env.example`

### Task 4: Run export on current top_songs; document VRAM/peft requirements
