import time
import os
import json
from typing import List, Dict, Any

import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI

# Ensure NLTK data
try:
	 nltk.data.find('tokenizers/punkt_tab')
except LookupError:
	 nltk.download('punkt_tab', quiet=True)
try:
	 nltk.data.find('tokenizers/punkt')
except LookupError:
	 nltk.download('punkt', quiet=True)

LABELS: List[str] = [
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
	"none",
]

SYSTEM_PROMPT = (
	"Classify each sentence into exactly one label from the allowed set. "
	"Use the full paragraph context. Only label a fallacy if a clear, explicit instance is present; "
	"otherwise return 'none'. Respond ONLY in compact JSON: results=[{index,label,confidence}]. "
	"Set confidence to a probability between 0 and 1."
)


def _build_user_msg(text: str, sentences: List[str]) -> str:
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


def _find_spans(text: str, sentences: List[str]) -> List[Dict[str, Any]]:
	spans = []
	pos = 0
	for s in sentences:
		idx = text.find(s, pos)
		if idx == -1:
			idx = pos
		start = idx
		end = start + len(s)
		spans.append({"start": start, "end": end, "text": s})
		pos = end
	return spans


def analyze_text(text: str, model_id: str, threshold: float = 0.6, max_tokens: int = 512) -> Dict[str, Any]:
	"""
	Analyze text with the fine-tuned OpenAI model using full-paragraph context and batching.
	Returns a dict: {
		input_text, elapsed_seconds, fallacies: [{fallacy_type, text, start_char, end_char, confidence}],
		fallacy_types: [...], sentences_with_fallacies: [...]
	}
	"""
	if not os.getenv('OPENAI_API_KEY'):
		raise RuntimeError("OPENAI_API_KEY is not set")

	client = OpenAI()

	try:
		sentences = sent_tokenize(text)
	except Exception:
		sentences = [s.strip() for s in text.split('.') if s.strip()]

	start_time = time.perf_counter()

	msg = client.chat.completions.create(
		model=model_id,
		temperature=0,
		max_tokens=max_tokens,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": _build_user_msg(text, sentences)}
		],
		response_format={"type": "json_object"}
	)
	content = msg.choices[0].message.content
	try:
		data = json.loads(content)
	except Exception:
		# Fallback to first JSON object
		start = content.find('{')
		end = content.rfind('}')
		if start != -1 and end != -1 and end > start:
			data = json.loads(content[start:end+1])
		else:
			raise

	results = data.get('results', [])
	spans = _find_spans(text, sentences)

	fallacies = []
	for i, span in enumerate(spans):
		label = 'none'
		conf = 0.0
		for item in results:
			try:
				idx = int(item.get('index', 0)) - 1
			except Exception:
				idx = -1
			if idx == i:
				candidate = str(item.get('label', '')).strip()
				try:
					conf = float(item.get('confidence', 0.0))
				except Exception:
					conf = 0.0
				label = candidate if candidate in LABELS else 'none'
				break
		if conf < threshold:
			label = 'none'
		fallacies.append({
			'fallacy_type': label,
			'text': span['text'],
			'start_char': span['start'],
			'end_char': span['end'],
			'confidence': round(conf, 4)
		})

	elapsed = time.perf_counter() - start_time

	sentences_with_fallacies = [f['text'] for f in fallacies if f['fallacy_type'] != 'none']
	fallacy_types = sorted({f['fallacy_type'] for f in fallacies if f['fallacy_type'] != 'none'})

	return {
		'input_text': text,
		'elapsed_seconds': elapsed,
		'fallacies': fallacies,
		'fallacy_types': fallacy_types,
		'sentences_with_fallacies': sentences_with_fallacies,
	}
