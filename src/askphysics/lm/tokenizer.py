"""Byte-level BPE tokenizer for the Fermi models, written from scratch.

- Base vocabulary: the 256 byte values, so any text encodes (byte fallback).
- Special tokens follow the bytes: ``<|pad|>``, ``<|end|>``, ``<|classify|>``,
  ``<|plan|>``, ``<|explain|>``.
- Learned merges fill the rest of the vocabulary.
- **Digits are never merged.** Pre-tokenization isolates every digit, so
  numbers are always spelled one digit per token. That lets the models copy
  numbers exactly and lets constrained decoding restrict them (``docs/MODELS.md``).
- Special-token strings inside user text are encoded as plain bytes unless
  the caller explicitly allows them, so a question containing ``<|plan|>``
  cannot switch tasks.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

PAD = "<|pad|>"
END = "<|end|>"
CLASSIFY = "<|classify|>"
PLAN = "<|plan|>"
EXPLAIN = "<|explain|>"
SPECIAL_TOKENS = (PAD, END, CLASSIFY, PLAN, EXPLAIN)

N_BYTES = 256
FILE_VERSION = 1

# Order matters: a digit is always its own chunk; a word may carry one leading space.
_PRETOKENIZE = re.compile(r"[0-9]| ?[A-Za-z]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s")
_SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

Pair = tuple[int, int]


def pretokenize(text: str) -> list[str]:
    """Split text into chunks that merges may not cross."""
    return _PRETOKENIZE.findall(text)


class Tokenizer:
    """A trained byte-level BPE tokenizer. Build one with ``train`` or ``load``."""

    def __init__(self, merges: Sequence[Pair]) -> None:
        self.merges: list[Pair] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.special_ids = {tok: N_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}
        first_merge_id = N_BYTES + len(SPECIAL_TOKENS)
        for i, (a, b) in enumerate(self.merges):
            if not (0 <= a < first_merge_id + i and 0 <= b < first_merge_id + i):
                raise ValueError(f"merge {i} ({a}, {b}) refers to a token that doesn't exist yet")
        if len(set(self.merges)) != len(self.merges):
            raise ValueError("duplicate merges")
        self._ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self._merge_ids: dict[Pair, int] = {
            pair: first_merge_id + i for i, pair in enumerate(self.merges)
        }
        self._bytes: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        for tok, i in self.special_ids.items():
            self._bytes[i] = tok.encode()
        for (a, b), i in self._merge_ids.items():
            self._bytes[i] = self._bytes[a] + self._bytes[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return N_BYTES + len(SPECIAL_TOKENS) + len(self.merges)

    @property
    def pad_id(self) -> int:
        return self.special_ids[PAD]

    @property
    def end_id(self) -> int:
        return self.special_ids[END]

    # ------------------------------------------------------------------ training

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> Tokenizer:
        """Learn merges from ``texts`` until the vocabulary reaches ``vocab_size``.

        Stops early if no pair occurs at least twice. Uses incremental pair
        counts, so each merge only touches the words that contain the pair.
        """
        n_merges = vocab_size - N_BYTES - len(SPECIAL_TOKENS)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {N_BYTES + len(SPECIAL_TOKENS)}")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            for part in _SPECIAL_SPLIT.split(text):
                if part and part not in SPECIAL_TOKENS:
                    chunk_counts.update(pretokenize(part))

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, count in chunk_counts.items():
            ids = list(chunk.encode())
            if len(ids) > 1 and not chunk.isdigit():
                words.append(ids)
                freqs.append(count)

        pair_counts: Counter[Pair] = Counter()
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for w, ids in enumerate(words):
            for pair in pairwise(ids):
                pair_counts[pair] += freqs[w]
                where[pair].add(w)

        merges: list[Pair] = []
        next_id = N_BYTES + len(SPECIAL_TOKENS)
        for _ in range(n_merges):
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < 2:
                break
            merges.append(best)
            for w in list(where[best]):
                ids = words[w]
                for pair in pairwise(ids):
                    pair_counts[pair] -= freqs[w]
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(w)
                merged: list[int] = []
                i = 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                        merged.append(next_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                words[w] = merged
                for pair in pairwise(merged):
                    pair_counts[pair] += freqs[w]
                    where[pair].add(w)
            where.pop(best, None)
            next_id += 1
        return cls(merges)

    # ------------------------------------------------------------------ encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode())
        while len(ids) > 1:
            pairs = list(pairwise(ids))
            best = min(pairs, key=lambda p: self._ranks.get(p, len(self._ranks)))
            if best not in self._ranks:
                break
            new_id = self._merge_ids[best]
            merged: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, *, allow_special: bool = False) -> list[int]:
        """Encode text to token ids.

        Args:
            allow_special: if False (the default, and the only safe choice for
                user text), special-token strings are encoded as plain bytes.
        """
        out: list[int] = []
        parts = _SPECIAL_SPLIT.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_ids:
                out.append(self.special_ids[part])
                continue
            for chunk in pretokenize(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back to text. Invalid UTF-8 is replaced, never raised."""
        try:
            data = b"".join(self._bytes[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from None
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes a token stands for (used by constrained decoding)."""
        return self._bytes[token_id]

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        payload = {
            "version": FILE_VERSION,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(m) for m in self.merges],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != FILE_VERSION:
            raise ValueError(f"unsupported tokenizer file version {payload.get('version')}")
        if tuple(payload.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("tokenizer special tokens do not match this version of askphysics")
        return cls([(int(a), int(b)) for a, b in payload["merges"]])
