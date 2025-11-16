# Fallac-Machine

A logical fallacy detector that now uses an OpenAI fine‑tuned model for sentence‑level classification. The detector analyzes input text, identifies fallacy types per sentence, and outputs results (including character spans) to JSON.

## Features

- Detects 13 different types of logical fallacies:
  - Ad hominem
  - Ad populum
  - Appeal to emotion
  - Circular reasoning
  - Equivocation
  - Fallacy of credibility
  - Fallacy of extension
  - Fallacy of logic
  - Fallacy of relevance
  - False causality
  - False dilemma
  - Faulty generalization
  - Intentional

- Provides character-level location information for detected fallacies
- Outputs results in JSON format

## Setup

1. Make sure you have Python 3.13 installed
2. Activate the virtual environment:
   ```powershell
   .\FMenv\Scripts\Activate.ps1
   ```
3. Set your OpenAI API key (PowerShell):
   ```powershell
   setx OPENAI_API_KEY "YOUR_KEY"    # persist
   $env:OPENAI_API_KEY = "YOUR_KEY"  # current session
   ```

## Training (OpenAI Supervised Fine‑Tuning)

Follow the concise guide:

- See `OPENAI_FINE_TUNE_GUIDE.md`
- In short:
  ```powershell
  .\FMenv\Scripts\python.exe scripts\prepare_openai_finetune.py --clean --outdir openai_ft
  .\FMenv\Scripts\python.exe scripts\fine_tune_openai.py --base-model <BASE_MODEL> --train openai_ft\train.jsonl --val openai_ft\val.jsonl
  ```
  Note: replace `<BASE_MODEL>` with your OpenAI base model id. After the job completes, the script prints a `fine_tuned_model` id.

## Using the Fine‑Tuned Model

### From Command Line

- Detect fallacies with your fine‑tuned model:
```powershell
.\FMenv\Scripts\python.exe detect_fallacies_openai.py --model <FINE_TUNED_MODEL_ID> --text "Your text here" --output results.json
```

- Using a text file:
```powershell
.\FMenv\Scripts\python.exe detect_fallacies_openai.py --model <FINE_TUNED_MODEL_ID> --file input.txt --output results.json
```

### Output Format

The JSON output contains:
```json
{
  "input_text": "The original input text",
  "total_sentences": 2,
  "fallacies": [
    {
      "fallacy_type": "ad populum",
      "text": "Since many people believe this, then it must be true.",
      "start_char": 0,
      "end_char": 63,
      "rationale": "... brief model rationale ..."
    }
  ]
}
```

## Notes

- The previous scikit‑learn training and detection scripts have been removed in favor of the OpenAI path.
- Keep your API key out of source control. Scripts read `OPENAI_API_KEY` from the environment.
- See `OPENAI_FINE_TUNE_GUIDE.md` for end‑to‑end steps (data prep, fine‑tune, detect).
