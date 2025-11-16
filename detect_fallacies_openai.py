import os
import sys
import json
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

LABELS = [
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "circular reasoning",
    "equivocation",
    "fallacy of credibility",
    "fallacy of extension",
    "fallacy of logic",
    "fallacy of relevance",
    "false causality",
    "false dilemma",
    "faulty generalization",
    "intentional",
    "miscellaneous",
    "none"
]

SYSTEM_PROMPT_COMPACT = (
    "Classify each sentence into exactly one label from the allowed set. "
    "Use the full paragraph context. Only label a fallacy if a clear, explicit instance is present; "
    "otherwise return 'none'. Respond ONLY in compact JSON: results=[{index,label,confidence}]. "
    "Set confidence to a probability between 0 and 1."
)


def build_batch_user_message(text: str, sentences: list[str]) -> str:
    allowed = ", ".join(LABELS)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return (
        f"Allowed labels: {allowed}.\n"
        f"Paragraph: {text}\n"
        f"Sentences (numbered):\n{numbered}\n\n"
        "Return JSON with array 'results', each item: {index, label, confidence}. "
        "Index is the 1-based sentence number; label is one of the allowed labels; "
        "confidence is a probability 0..1 for the chosen label."
    )


def classify_batch(client: OpenAI, model: str, text: str, sentences: list[str]) -> list[dict]:
    msg = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=512,  # allow larger JSON for many sentences
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_COMPACT},
            {"role": "user", "content": build_batch_user_message(text, sentences)}
        ],
        response_format={"type": "json_object"}
    )
    content = msg.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # Fallback: extract first JSON object substring
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start:end+1])
        else:
            raise
    results = data.get('results', [])
    norm = []
    for item in results:
        try:
            idx = int(item.get('index', 0)) - 1
        except Exception:
            idx = -1
        label = str(item.get('label', '')).strip()
        try:
            conf = float(item.get('confidence', 0))
        except Exception:
            conf = 0.0
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0
        if 0 <= idx < len(sentences) and label in LABELS:
            norm.append({'index': idx, 'label': label, 'confidence': conf})
    return norm


def find_fallacy_spans(text: str, sentences: list[str]):
    spans = []
    current_pos = 0
    for sentence in sentences:
        idx = text.find(sentence, current_pos)
        if idx == -1:
            idx = current_pos
        start = idx
        end = start + len(sentence)
        spans.append({'start': start, 'end': end, 'text': sentence})
        current_pos = end
    return spans


def main():
    parser = argparse.ArgumentParser(description='Detect fallacies using a fine-tuned OpenAI model (context-aware batch)')
    parser.add_argument('--model', required=True, help='Fine-tuned OpenAI model id/name')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='Input file path')
    parser.add_argument('--output', type=str, default='output_openai.json', help='Output JSON path')
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        print('Error: OPENAI_API_KEY is not set in environment.')
        sys.exit(1)

    client = OpenAI()

    # Read input
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    text = (text or '').strip()
    if not text:
        print('Error: No input text provided.')
        sys.exit(1)

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split('.') if s.strip()]

    if not sentences:
        out = {'input_text': text, 'total_sentences': 0, 'fallacies': []}
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'Saved results to {args.output}')
        return

    # Batch classify using full context
    batch = classify_batch(client, args.model, text, sentences)
    spans = find_fallacy_spans(text, sentences)

    results = []
    for i, span in enumerate(spans):
        label = 'none'
        confidence = 0.0
        for item in batch:
            if item['index'] == i:
                label = item['label']
                confidence = float(item.get('confidence', 0.0))
                break
        results.append({
            'fallacy_type': label,
            'text': span['text'],
            'start_char': span['start'],
            'end_char': span['end'],
            'confidence': round(confidence, 4)
        })

    out = {
        'input_text': text,
        'total_sentences': len(sentences),
        'fallacies': results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main()
