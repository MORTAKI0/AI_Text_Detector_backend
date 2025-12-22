import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "Torch is required to start the API. Install with "
        "pip install \"torch>=2.9.0,<2.10\"."
    ) from exc
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .settings import get_settings

settings = get_settings()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGMENT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def get_model_components():
    model_path = os.path.normpath(os.path.abspath(settings.model_dir))
    tok_kwargs = {
        "use_fast": True,
        "local_files_only": True,
        "fix_mistral_regex": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    except TypeError:
        tok_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    model.to(device)
    model.eval()
    model_name = model.config._name_or_path or "model"
    return tokenizer, model, model_name


def _predict_proba(text: str) -> float:
    tokenizer, model, _ = get_model_components()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    return float(probs[1])


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    last_idx = 0
    for match in SEGMENT_PATTERN.finditer(text):
        end_idx = match.start()
        spans.append((last_idx, end_idx))
        last_idx = match.end()
    spans.append((last_idx, len(text)))
    return spans


def analyze_text(
    text: str, top_k: int = 8, suspicious_threshold: float = 0.60
) -> Dict:
    tokenizer, model, model_name = get_model_components()
    _ = tokenizer  # silence unused linting in some editors
    threshold = settings.AI_THRESHOLD
    spans = _sentence_spans(text)

    scored_segments: List[Dict] = []
    for start_idx, end_idx in spans:
        raw_segment = text[start_idx:end_idx]
        stripped = raw_segment.strip()
        if not stripped:
            continue
        leading_trim = len(raw_segment) - len(raw_segment.lstrip())
        trailing_trim = len(raw_segment.rstrip())
        adj_start = start_idx + leading_trim
        adj_end = start_idx + trailing_trim
        if adj_start >= adj_end:
            continue
        word_count = len(re.findall(r"\w+", stripped))
        if len(stripped) < 20 or word_count < 8:
            continue
        prob_ai = _predict_proba(stripped)
        scored_segments.append(
            {"start": adj_start, "end": adj_end, "prob_ai": prob_ai}
        )

    if not scored_segments:
        global_prob = _predict_proba(text)
        label = 1 if global_prob >= threshold else 0
        suspicious_segments = (
            [{"start": 0, "end": len(text), "prob_ai": global_prob}]
            if global_prob >= suspicious_threshold
            else []
        )
        return {
            "label": label,
            "prob_ai": global_prob,
            "segments": suspicious_segments,
            "model_name": model_name,
        }

    sorted_segments = sorted(
        scored_segments, key=lambda s: s["prob_ai"], reverse=True
    )
    global_prob = _predict_proba(text)
    label = 1 if global_prob >= threshold else 0

    suspicious_segments = [
        seg for seg in sorted_segments if seg["prob_ai"] >= suspicious_threshold
    ]
    if top_k > 0:
        suspicious_segments = suspicious_segments[:top_k]
    suspicious_segments.sort(key=lambda s: s["start"])

    return {
        "label": label,
        "prob_ai": global_prob,
        "segments": suspicious_segments,
        "model_name": model_name,
    }
