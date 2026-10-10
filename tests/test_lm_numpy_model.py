"""The numpy forward pass matches torch, and the decoder gives the same text over either (ADR-021).

No weights and no GPU: the models are randomly initialised ``fermi-luna-1`` (and a model of
tellus's shape) on CPU. Initial weights give near-uniform logits, so they are scaled up to make
the decoder's choices depend on the model.
"""

import json
import os
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import askphysics
from askphysics.data.loader import DataStore
from askphysics.errors import AskPhysicsError, ConfigError, LLMError
from askphysics.lm.checkpoints import load_model, model_metadata, save_model
from askphysics.lm.config import LUNA, TELLUS, ModelConfig
from askphysics.lm.engine import Cache, Engine, Logits
from askphysics.lm.formats import classify_prompt, explain_numbers, extract_numbers, plan_prompt
from askphysics.lm.generate import (
    Decoder,
    decode_classification,
    decode_explanation,
    decode_plan,
)
from askphysics.lm.model import FermiLM
from askphysics.lm.numpy_model import (
    NumpyFermiLM,
    load_numpy_model,
    parameter_shapes,
    read_safetensors,
)
from askphysics.lm.package import to_bf16
from askphysics.lm.paths import TOKENIZER_FILE, WEIGHTS_FILE
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.torch_engine import TorchEngine
from askphysics.models import Constant, Equation
from askphysics.retrieval.keyword import KeywordRetriever
from askphysics.solver.units import quantity

QUESTIONS = [
    "How fast does a ball dropped from 20 m hit the ground?",
    "A 2 kg block is pushed with a 10 N force. What is its acceleration?",
    "How many piano tuners are there in Chicago?",
    "What temperature is anger?",
    "A car goes from 0 to 27 m/s in 6 s. How far does it travel?",
]

# A model with tellus's shape (about 3M parameters), for parity at a realistic width and depth.
TELLUS_SHAPED = ModelConfig(
    name="tellus-shaped",
    vocab_size=TELLUS.vocab_size,
    d_model=TELLUS.d_model,
    n_layers=TELLUS.n_layers,
    n_heads=TELLUS.n_heads,
    context_length=TELLUS.context_length,
)


@pytest.fixture(scope="module")
def tokenizer(store: DataStore) -> Tokenizer:
    consts = list(store.constants.values())
    texts = [plan_prompt(QUESTIONS[0], "standard", [eq], consts) for eq in store.equations.values()]
    texts += [
        classify_prompt(QUESTIONS[0]),
        "standard fermi out_of_scope given constant assumption",
    ]
    return Tokenizer.train(texts * 3, vocab_size=LUNA.vocab_size)


def _random_model(config: ModelConfig, seed: int, scale: float) -> FermiLM:
    torch.manual_seed(seed)
    model = FermiLM(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    return model


def _saved(
    tmp_path: Path, tokenizer: Tokenizer, config: ModelConfig, seed: int, scale: float
) -> Path:
    save_model(_random_model(config, seed, scale), tokenizer, tmp_path / "model")
    return tmp_path / "model"


# ------------------------------------------------------------------ the forward pass


@pytest.mark.parametrize(
    ("config", "scale", "tokens", "atol"),
    [(LUNA, 3.0, 40, 5e-5), (TELLUS_SHAPED, 2.0, 96, 1e-4)],
    ids=["luna", "tellus-shaped"],
)
def test_logits_match_torch_with_and_without_the_cache(
    tmp_path: Path,
    tokenizer: Tokenizer,
    config: ModelConfig,
    scale: float,
    tokens: int,
    atol: float,
) -> None:
    directory = _saved(tmp_path, tokenizer, config, seed=0, scale=scale)
    torch_model, _ = load_model(directory)
    numpy_model, _ = load_numpy_model(directory)
    ids = torch.randint(0, tokenizer.vocab_size, (2, tokens), generator=torch.manual_seed(1))

    expected, _ = torch_model(ids)  # no cache: the whole sequence at once
    actual, _ = numpy_model.step(ids.numpy())
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=0, atol=atol)

    split = tokens // 2  # cached: a prompt, then one token per step
    logits, past = numpy_model.step(ids.numpy()[:, :split])
    steps = [logits]
    for t in range(split, tokens):
        logits, past = numpy_model.step(ids.numpy()[:, t : t + 1], past)
        steps.append(logits)
    np.testing.assert_allclose(
        np.concatenate(steps, axis=1), expected.detach().numpy(), rtol=0, atol=atol
    )
    torch_logits, torch_past = torch_model.step(ids[:, :split])
    np.testing.assert_allclose(steps[0], torch_logits.numpy(), rtol=0, atol=atol)
    assert past[0][0].shape == (2, config.n_heads, tokens, config.head_dim)
    assert len(torch_past) == len(past) == config.n_layers


def test_bf16_weights_load_exactly_as_torch_casts_them(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    source = _saved(tmp_path, tokenizer, LUNA, seed=2, scale=3.0)
    to_bf16(source, tmp_path / "bf16")
    assert (tmp_path / "bf16" / WEIGHTS_FILE).stat().st_size < (
        source / WEIGHTS_FILE
    ).stat().st_size
    torch_model, _ = load_model(tmp_path / "bf16")
    numpy_model, _ = load_numpy_model(tmp_path / "bf16")
    torch_weights = torch_model.state_dict()
    tensors, _ = read_safetensors((tmp_path / "bf16" / WEIGHTS_FILE).read_bytes())
    assert set(tensors) == set(torch_weights)
    for name, array in tensors.items():
        assert array.dtype == np.float32
        np.testing.assert_array_equal(array, torch_weights[name].numpy(), err_msg=name)
    ids = np.arange(24).reshape(1, 24) % tokenizer.vocab_size
    np.testing.assert_allclose(
        numpy_model.step(ids)[0],
        torch_model(torch.from_numpy(ids))[0].detach().numpy(),
        rtol=0,
        atol=5e-5,
    )


def test_the_cache_takes_one_token_at_a_time_and_respects_the_context(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    model, _ = load_numpy_model(_saved(tmp_path, tokenizer, LUNA, seed=0, scale=1.0))
    _, past = model.step(np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="one token"):
        model.step(np.zeros((1, 2), dtype=np.int64), past)
    with pytest.raises(ValueError, match="exceeds context"):
        model.step(np.zeros((1, LUNA.context_length + 1), dtype=np.int64))
    with pytest.raises(ValueError, match="nothing to feed"):
        model.next_logits([], None)


def test_a_truncated_cache_leaves_the_original_valid(tmp_path: Path, tokenizer: Tokenizer) -> None:
    model, _ = load_numpy_model(_saved(tmp_path, tokenizer, LUNA, seed=0, scale=3.0))
    tokens = [5, 9, 3, 7, 11, 2]
    logits, past = model.next_logits(tokens, None)
    short = model.truncate(past, 3)
    assert model.cache_length(short) == 3 and model.cache_length(past) == 6
    # Re-feeding the dropped tokens from the short cache gives the same logits again.
    again = short
    for t in tokens[3:]:
        again_logits, again = model.next_logits([t], again)
    np.testing.assert_allclose(again_logits, logits, rtol=0, atol=5e-5)
    assert model.cache_length(past) == 6


# ------------------------------------------------------------------ weights files


def _safetensors(header: dict[str, Any], body: bytes = b"") -> bytes:
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + body


def test_the_reader_reads_every_stored_dtype(tmp_path: Path) -> None:
    tensors = {"a": torch.tensor([[1.5, -2.0, 0.1]]), "b": torch.arange(4.0)}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        path = tmp_path / f"{dtype}.safetensors"
        save_file({k: v.to(dtype) for k, v in tensors.items()}, str(path), metadata={"k": "v"})
        read, metadata = read_safetensors(path.read_bytes())
        assert metadata == {"k": "v"}
        for name, tensor in tensors.items():
            assert read[name].dtype == np.float32 and read[name].shape == tuple(tensor.shape)
            np.testing.assert_array_equal(read[name], tensor.to(dtype).float().numpy())


def test_the_reader_copies_so_the_file_can_be_freed() -> None:
    header = {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    data = bytearray(_safetensors(header, struct.pack("<HH", 0x3F80, 0xC000)))
    tensors, _ = read_safetensors(data)
    data[:] = b""  # a bytearray with live views cannot be resized, so none are held
    assert tensors["w"].tolist() == [1.0, -2.0]


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"short", "length prefix"),
        (struct.pack("<Q", 10**9) + b"{}", "header length"),
        (struct.pack("<Q", 3) + b"{no", "not JSON"),
        (
            _safetensors({"w": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}, bytes(8)),
            "'w'",
        ),
        (
            _safetensors({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, bytes(8)),
            "'w'",
        ),
        (
            _safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, bytes(4)),
            "'w'",
        ),
        (_safetensors({"w": {"dtype": "F32"}}), "'w'"),
    ],
    ids=["short", "huge-header", "not-json", "dtype", "shape", "truncated", "missing-keys"],
)
def test_the_reader_refuses_malformed_files(data: bytes, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        read_safetensors(data)


def test_missing_or_misshapen_weights_are_a_config_error() -> None:
    weights = {n: np.zeros(s, dtype=np.float32) for n, s in parameter_shapes(LUNA).items()}
    NumpyFermiLM(LUNA, weights)
    with pytest.raises(ConfigError, match=r"missing norm\.weight"):
        NumpyFermiLM(LUNA, {k: v for k, v in weights.items() if k != "norm.weight"})
    weights["embed.weight"] = np.zeros((LUNA.vocab_size, LUNA.d_model + 1), dtype=np.float32)
    with pytest.raises(ConfigError, match=r"embed\.weight has shape"):
        NumpyFermiLM(LUNA, weights)


def test_the_parameter_names_are_the_torch_ones() -> None:
    names = {n: tuple(p.shape) for n, p in FermiLM(LUNA).named_parameters()}
    assert parameter_shapes(LUNA) == names


def test_loading_refuses_what_the_torch_loader_refuses(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    directory = _saved(tmp_path, tokenizer, LUNA, seed=0, scale=1.0)
    config = json.loads((directory / "config.json").read_text())
    config["n_layers"] = 3
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConfigError, match="does not match"):
        load_numpy_model(directory)
    (directory / TOKENIZER_FILE).unlink()
    with pytest.raises(ConfigError, match=r"missing tokenizer\.json"):
        load_numpy_model(directory)


def test_a_model_from_another_task_format_is_refused(tmp_path: Path, tokenizer: Tokenizer) -> None:
    directory = _saved(tmp_path, tokenizer, LUNA, seed=0, scale=1.0)
    state = {
        k: v.detach().contiguous() for k, v in _random_model(LUNA, 0, 1.0).state_dict().items()
    }
    metadata = {**model_metadata(LUNA), "format_version": "999"}
    save_file(state, str(directory / WEIGHTS_FILE), metadata=metadata)
    with pytest.raises(ConfigError, match="task format"):
        load_numpy_model(directory)


# ------------------------------------------------------------------ decoding


def _attempt(run: Callable[[], object]) -> object:
    try:
        return run()
    except AskPhysicsError as exc:  # a slot with no legal value fails, as designed
        return f"{type(exc).__name__}: {exc}"


def _decode_question(
    decoder: Decoder,
    explainer: Decoder,
    question: str,
    eqs: list[Equation],
    consts: list[Constant],
) -> dict[str, object]:
    return {
        "classify": _attempt(lambda: decode_classification(decoder, question).model_dump()),
        "plan": _attempt(
            lambda: decode_plan(decoder, question, "standard", eqs, consts).model_dump()
        ),
        "explain": _attempt(
            lambda: decode_explanation(
                explainer, question, quantity(19.8057, "m/s"), eqs[:1], ["No drag"]
            )
        ),
    }


def _decode_all(
    decoder: Decoder, store: DataStore, explainer: Decoder
) -> dict[tuple[str, str], object]:
    """Classify, plan, and explain each question; a failure counts as a result too."""
    consts = list(store.constants.values())
    retriever = KeywordRetriever(store.equations.values(), store.examples.values())
    results: dict[tuple[str, str], object] = {}
    for question in QUESTIONS:
        eqs = [hit.equation for hit in retriever.search(question, 3).equations]
        for kind, value in _decode_question(decoder, explainer, question, eqs, consts).items():
            results[question, kind] = value
    return results


class _EndsSoon(Engine):
    """An engine that starts favouring the end token after a few steps, so prose finishes."""

    def __init__(self, inner: Engine, end_id: int, after: int = 20) -> None:
        self.inner, self.end_id, self.after, self.calls = inner, end_id, after, 0
        self.config, self.device = inner.config, inner.device

    def next_logits(self, tokens: Sequence[int], past: Cache | None) -> tuple[Logits, Cache]:
        logits, new_past = self.inner.next_logits(tokens, past)
        self.calls += 1
        if self.calls > self.after:
            logits = logits.copy()
            logits[self.end_id] += 1000.0
        return logits, new_past

    def generator(self, seed: int) -> Any:
        return self.inner.generator(seed)

    def sample(self, masked: Logits, temperature: float, generator: Any) -> int:
        return self.inner.sample(masked, temperature, generator)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classify_plan_and_explain_are_identical_on_both_backends(
    tmp_path: Path, store: DataStore, tokenizer: Tokenizer, seed: int
) -> None:
    directory = _saved(tmp_path, tokenizer, LUNA, seed=seed, scale=3.0)
    torch_model, _ = load_model(directory)
    numpy_model, _ = load_numpy_model(directory)
    outputs = []
    for engine in (TorchEngine(torch_model), numpy_model):
        decoder = Decoder(engine, tokenizer, max_slot_tokens=24)
        explainer = Decoder(_EndsSoon(engine, tokenizer.end_id), tokenizer)
        outputs.append(_decode_all(decoder, store, explainer))
    assert outputs[0] == outputs[1]
    # The comparison is not between two sets of error messages: plans and prose were written.
    plans = [v for (_, kind), v in outputs[0].items() if kind == "plan" and isinstance(v, dict)]
    prose = [v for (_, kind), v in outputs[0].items() if kind == "explain" and v]
    assert plans and any(isinstance(v, str) and "Error" not in v for v in prose)


def test_a_torch_model_and_a_numpy_model_are_both_accepted_by_the_decoder(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    directory = _saved(tmp_path, tokenizer, LUNA, seed=0, scale=3.0)
    torch_model, _ = load_model(directory)
    numpy_model, _ = load_numpy_model(directory)
    on_torch, on_numpy = Decoder(torch_model, tokenizer), Decoder(numpy_model, tokenizer)
    assert on_torch.engine.device == "cpu" and on_numpy.engine is numpy_model
    assert on_torch.model is torch_model and on_numpy.model is numpy_model
    assert decode_classification(on_torch, QUESTIONS[0]) == decode_classification(
        on_numpy, QUESTIONS[0]
    )


def test_sampling_is_seeded_and_cannot_pick_a_masked_token(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    model, _ = load_numpy_model(_saved(tmp_path, tokenizer, LUNA, seed=0, scale=3.0))
    masked = np.full(LUNA.vocab_size, -np.inf, dtype=np.float32)
    masked[[7, 8, 9]] = [0.0, 1.0, 2.0]
    draws = [model.sample(masked, 1.0, model.generator(seed)) for seed in range(40)]
    assert set(draws) <= {7, 8, 9} and len(set(draws)) > 1
    assert draws == [model.sample(masked, 1.0, model.generator(seed)) for seed in range(40)]
    with pytest.raises(LLMError, match="no token"):
        model.sample(np.full(LUNA.vocab_size, -np.inf, dtype=np.float32), 1.0, model.generator(0))


def test_a_sampled_explanation_on_numpy_only_spells_numbers_from_its_input(
    tmp_path: Path, store: DataStore, tokenizer: Tokenizer
) -> None:
    model, _ = load_numpy_model(_saved(tmp_path, tokenizer, LUNA, seed=1, scale=3.0))
    decoder = Decoder(_EndsSoon(model, tokenizer.end_id), tokenizer)
    eqs = [store.equations["kin_v_squared"]]
    result = quantity(19.8057, "m/s")
    text = decode_explanation(
        decoder, QUESTIONS[0], result, eqs, ["No drag"], temperature=1.0, seed=3
    )
    assert set(extract_numbers(text)) <= set(explain_numbers(QUESTIONS[0], result, ["No drag"]))


# ------------------------------------------------------------------ no torch


PROBE = """
import sys
sys.modules["torch"] = None  # Pyodide has no torch
sys.modules["safetensors"] = None  # and the numpy path does not need the package
from pathlib import Path
from askphysics.data.loader import load_all
from askphysics.lm.generate import Decoder, decode_classification, decode_plan
from askphysics.lm.numpy_model import load_numpy_model
from askphysics.lm.formats import relevant_constants

model, tokenizer = load_numpy_model(Path(sys.argv[1]))
decoder = Decoder(model, tokenizer, max_slot_tokens=24)
question = sys.argv[2]
print("classified", decode_classification(decoder, question).category)
store = load_all()
eqs = [store.equations["kin_v_squared"]]
consts = relevant_constants(eqs, list(store.constants.values()))
try:
    decode_plan(decoder, question, "standard", eqs, consts)
except Exception as exc:
    print("plan", type(exc).__name__)
else:
    print("plan ok")
assert "torch" not in {m for m in sys.modules if sys.modules[m] is not None}
"""


def test_the_numpy_path_imports_no_torch(tmp_path: Path, tokenizer: Tokenizer) -> None:
    directory = _saved(tmp_path, tokenizer, LUNA, seed=0, scale=3.0)
    source = str(Path(askphysics.__file__).parent.parent)  # this checkout, not an installed copy
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(directory), QUESTIONS[0]],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": source},
    )
    assert result.returncode == 0, result.stderr
    assert "classified" in result.stdout and "plan" in result.stdout
