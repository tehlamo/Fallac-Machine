# OpenAI Supervised Fine-Tuning (SFT) Setup

This project supports supervised fine-tuning on an OpenAI base model for sentence-level fallacy classification.

## 1) Prerequisites

- Set your API key in the environment (PowerShell):

```powershell
setx OPENAI_API_KEY "YOUR_KEY_HERE"
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"  # current session
```

- Choose your base model (e.g., `gpt-4o-mini`, or your provided project-scoped model id)

## 2) Prepare Training/Validation JSONL

```powershell
.\FMenv\Scripts\python.exe scripts\prepare_openai_finetune.py --clean --outdir openai_ft
```

This produces:
- `openai_ft\train.jsonl`
- `openai_ft\val.jsonl`

Format: each line contains `{ "prompt": "...", "completion": "<label>" }` for one sentence.

## 3) Start Fine-Tuning

```powershell
.\FMenv\Scripts\python.exe scripts\fine_tune_openai.py --base-model <YOUR_BASE_MODEL> ^
    --train openai_ft\train.jsonl --val openai_ft\val.jsonl
```

- The script uploads both files, starts the job, and prints a job id.
- It waits for completion and prints the fine-tuned model id on success.

## 4) Run Detection with the Fine-Tuned Model

```powershell
.\FMenv\Scripts\python.exe detect_fallacies_openai.py --model <FINE_TUNED_MODEL_ID> ^
    --text "Are you stupid? Don't you care about the people?" --output openai_result.json
```

- Output JSON includes each sentence’s predicted label, character spans, and model rationale.

## Notes

- Your API key is never written to disk by these scripts; keep it in environment variables.
- You can pass the provided model id from your message as `--base-model` when creating a job, and later as `--model` when detecting.
- Training data is sentence-level; if you want multi-label per document later, we can extend the format.
