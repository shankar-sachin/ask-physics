"""Real prose for a language-modeling stage before task training (ADR-016).

Everything else the Fermi models read was written by our data factory, so they
never see real English. ``third_party/openstax-physics/prose.jsonl`` holds the
body text of OpenStax *Physics* (CC BY 4.0, see its ``ATTRIBUTION.md``), one
paragraph per line. It trains as plain next-token prediction: no prompt, every
token after the first is a target, and the paragraph ends with the end token.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from askphysics.lm.generate import encode_task
from askphysics.lm.tokenizer import END, Tokenizer
from askphysics.lm.train import TokenizedSet

PROSE_TASK = "prose"
VAL_EVERY = 20  # every 20th paragraph is held out for val_loss_prose


def read_prose(path: Path) -> list[str]:
    """Paragraph texts from a ``prose.jsonl`` written by ``scripts/extract_openstax.py``."""
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            text = str(json.loads(line)["text"]).strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"no paragraphs in {path}")
    return texts


def split_prose(texts: Sequence[str]) -> tuple[list[str], list[str]]:
    """(train, val): every ``VAL_EVERY``-th paragraph is held out."""
    train = [t for i, t in enumerate(texts) if i % VAL_EVERY != VAL_EVERY - 1]
    val = [t for i, t in enumerate(texts) if i % VAL_EVERY == VAL_EVERY - 1]
    return train, val


def tokenize_prose(texts: Sequence[str], tokenizer: Tokenizer, context_length: int) -> TokenizedSet:
    """Paragraphs as training rows; a paragraph longer than the context is split into chunks."""
    out = TokenizedSet()
    for text in texts:
        ids = encode_task(tokenizer, text + END)
        for start in range(0, len(ids), context_length):
            chunk = ids[start : start + context_length + 1]
            if len(chunk) >= 2:
                out.rows.append((chunk, 1))  # no prompt: learn every token after the first
                out.tasks.append(PROSE_TASK)
    return out
