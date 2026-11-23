"""Tests for ``nousergon_lib.rag.embeddings`` with the Voyage client stubbed.

Nothing here touches a model or the network: ``_client`` is replaced with a
recording fake, and ``_get_client`` is exercised against a fake ``voyageai``
module and a stubbed secret lookup.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from nousergon_lib.rag import embeddings


@dataclass
class _Result:
    embeddings: list[list[float]]


@dataclass
class _FakeClient:
    calls: list[tuple[list[str], str, str]] = field(default_factory=lambda: [])

    def embed(self, texts: list[str], *, model: str, input_type: str) -> _Result:
        self.calls.append((list(texts), model, input_type))
        return _Result([[float(len(t))] for t in texts])


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(embeddings, "_client", client)
    return client


def test_embed_texts_returns_vectors_in_input_order(fake_client: _FakeClient) -> None:
    assert embeddings.embed_texts(["a", "bbb", "cc"]) == [[1.0], [3.0], [2.0]]
    assert fake_client.calls == [(["a", "bbb", "cc"], "voyage-3-lite", "document")]


def test_embed_texts_batches_at_batch_size(fake_client: _FakeClient) -> None:
    out = embeddings.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2)
    assert [c[0] for c in fake_client.calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    # Mutation guarded: dropping the slice step would repeat or lose a batch.
    assert out == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_embed_texts_passes_model_and_input_type(fake_client: _FakeClient) -> None:
    embeddings.embed_texts(["x"], input_type="query", model="example-model")
    assert fake_client.calls == [(["x"], "example-model", "query")]


def test_embed_texts_empty_input_makes_no_call(fake_client: _FakeClient) -> None:
    assert embeddings.embed_texts([]) == []
    assert fake_client.calls == []


def test_embed_query_uses_query_input_type_and_unwraps(fake_client: _FakeClient) -> None:
    assert embeddings.embed_query("hello") == [5.0]
    assert fake_client.calls == [(["hello"], "voyage-3-lite", "query")]


def test_embed_query_passes_model_through(fake_client: _FakeClient) -> None:
    embeddings.embed_query("hi", model="example-model")
    assert fake_client.calls[0][1] == "example-model"


def test_client_error_propagates_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def embed(self, texts: list[str], *, model: str, input_type: str) -> _Result:
            raise RuntimeError("upstream down")

    monkeypatch.setattr(embeddings, "_client", _Boom())
    with pytest.raises(RuntimeError, match="upstream down"):
        embeddings.embed_texts(["a"])


def test_get_client_builds_once_with_resolved_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str | None] = []

    class _VoyageClient:
        def __init__(self, api_key: str | None = None) -> None:
            built.append(api_key)

    fake_voyage = types.SimpleNamespace(Client=_VoyageClient)
    monkeypatch.setitem(sys.modules, "voyageai", fake_voyage)

    requested: list[tuple[str, bool]] = []

    def _get_secret(name: str, required: bool = True) -> str:
        requested.append((name, required))
        return "example-key"

    import krepis.secrets as secrets_module

    monkeypatch.setattr(secrets_module, "get_secret", _get_secret)
    monkeypatch.setattr(embeddings, "_client", None)

    first = embeddings._get_client()  # pyright: ignore[reportPrivateUsage]
    second = embeddings._get_client()  # pyright: ignore[reportPrivateUsage]
    assert first is second
    assert built == ["example-key"]
    assert requested == [("VOYAGE_API_KEY", False)]
