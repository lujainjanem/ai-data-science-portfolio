"""Optional feature: how *predictable* the text is to a small language model (GPT-2).

LLMs tend to pick high-probability next words, so their text gets a higher average
log-probability (lower perplexity) under another language model, and the per-token
scores vary less. Runs locally; needs `transformers` + `torch` (~500 MB download once).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PERPLEXITY_FEATURES = ["lm_logprob_mean", "lm_logprob_std"]


class PerplexityScorer:
    def __init__(self, model_name: str = "gpt2", max_tokens: int = 256, cache_file: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.max_tokens = max_tokens
        self.cache_file = cache_file
        self.cache: dict[str, list[float]] = {}
        if cache_file and cache_file.exists():
            self.cache = json.loads(cache_file.read_text())

    def score(self, text: str) -> dict[str, float]:
        key = hashlib.sha1(text.encode()).hexdigest()
        if key not in self.cache:
            ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens).input_ids
            with self.torch.no_grad():
                logits = self.model(ids).logits[0, :-1]
            logprobs = self.torch.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(1, ids[0, 1:, None]).squeeze(1).numpy()
            self.cache[key] = [float(np.mean(token_logprobs)), float(np.std(token_logprobs))]
        return dict(zip(PERPLEXITY_FEATURES, self.cache[key]))

    def save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache))
