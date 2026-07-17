"""Contract test: krepis.llm_capture's SFT v3 envelope stays in lockstep with
nousergon_lib.sft — the AGPL schema authority (alpha-engine-config#1660).

``krepis.llm_capture`` (public MIT, ``nousergon/krepis``) mirrors the
canonical SFT v3 record envelope owned by ``nousergon_lib.sft`` (this repo,
AGPL) field-for-field, because MIT products cannot depend on an AGPL
library. krepis IS already a base dependency of nousergon-lib (the inverse
direction is fine — AGPL depending on MIT — see the v0.66.0 relocation
covered by ``test_krepis_reexport.py``), so this repo's own test suite can
import both sides directly with no extra install wiring: no new dev-extra,
no vendored fixture, no cross-repo CI plumbing. That is the trigger for
this test living HERE rather than in krepis: nousergon_lib is the schema
authority AND the only one of the two repos that can hold both imports at
once without an AGPL leak.

A mirrored-by-hand schema drifts silently the moment one repo's envelope
changes without the other. This test is the structural guard (same
``test_..._pin_lockstep`` doctrine as ``test_version_pin.py``: two
independently-maintained sources of truth, pinned together so a one-sided
edit fails CI instead of shipping a corpus with two incompatible SFT
dialects). It fails on any of:

- a ``SFT_SCHEMA_VERSION`` constant bump on one side without the other;
- a ``krepis.llm_capture.build_sft_record`` output that no longer validates
  against ``nousergon_lib.sft.SftRecord`` (``extra="forbid"`` — a renamed /
  added / dropped field fails validation, not silently passes through);
- a ``content_hash`` algorithm divergence on a fixed input — the two must
  produce byte-identical dedup keys so a krepis-captured LIVE trace and a
  fleet REPLAY of the same call collapse under ``nousergon_lib.sft.dedup``.
"""

from __future__ import annotations

from krepis.llm import LLMResult, LLMUsage
from krepis.llm_capture import SFT_SCHEMA_VERSION as KREPIS_SFT_SCHEMA_VERSION
from krepis.llm_capture import build_sft_record
from krepis.llm_capture import content_hash as krepis_content_hash
from pydantic import ValidationError

from nousergon_lib.sft import SFT_SCHEMA_VERSION as NOUSERGON_LIB_SFT_SCHEMA_VERSION
from nousergon_lib.sft import SftRecord
from nousergon_lib.sft import content_hash as nousergon_lib_content_hash


def _synthetic_llm_result() -> LLMResult:
    """A minimal but representative ``krepis.llm.LLMResult`` — the input
    ``build_sft_record`` maps into the canonical envelope."""
    return LLMResult(
        text="the answer",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        usage=LLMUsage(input_tokens=100, output_tokens=50),
        raw_request={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "You are a judge."}],
            "messages": [{"role": "user", "content": "score this"}],
        },
        raw_response=None,
    )


def test_schema_version_constants_match():
    """Lockstep pin: bumping the envelope schema on one side without the
    other fails here. If you're updating this, you almost certainly need to
    bump BOTH ``krepis/src/krepis/llm_capture.py::SFT_SCHEMA_VERSION`` and
    ``src/nousergon_lib/sft.py::SFT_SCHEMA_VERSION`` in the same release
    window — search both repos for ``SFT_SCHEMA_VERSION``."""
    assert KREPIS_SFT_SCHEMA_VERSION == NOUSERGON_LIB_SFT_SCHEMA_VERSION, (
        f"krepis.llm_capture.SFT_SCHEMA_VERSION={KREPIS_SFT_SCHEMA_VERSION!r} "
        f"!= nousergon_lib.sft.SFT_SCHEMA_VERSION="
        f"{NOUSERGON_LIB_SFT_SCHEMA_VERSION!r} — the two envelopes have "
        "drifted. nousergon_lib.sft is the schema authority; bump krepis's "
        "mirror (and the field-for-field shape below) to match."
    )


def test_krepis_built_record_validates_against_nousergon_lib_schema():
    """A record ``krepis.llm_capture.build_sft_record`` emits must validate
    cleanly against ``nousergon_lib.sft.SftRecord`` — the AGPL side is the
    schema authority. ``SftRecord`` forbids extra fields, so a renamed,
    added, or dropped field on the krepis side fails here with a
    ``ValidationError`` rather than silently shipping a divergent record."""
    record = build_sft_record(
        _synthetic_llm_result(),
        producer="mnemon_judge",
        meta={"memory_id": 42},
        cost_usd=0.001,
        call_seq=1,
    )
    try:
        validated = SftRecord.model_validate(record)
    except ValidationError as exc:  # pragma: no cover - failure path
        raise AssertionError(
            "krepis.llm_capture.build_sft_record's output no longer "
            f"validates against nousergon_lib.sft.SftRecord: {exc}"
        ) from exc

    assert validated.producer == "mnemon_judge"
    assert validated.model == "claude-haiku-4-5-20251001"
    assert validated.provenance is not None
    assert validated.provenance.source == "live"


def test_content_hash_parity_on_fixed_input():
    """Both algorithms must produce a byte-identical dedup key on the SAME
    input so a krepis-captured LIVE trace and a fleet REPLAY mint of the
    same teacher call collapse under ``nousergon_lib.sft.dedup``."""
    fixed_input_messages = [
        {"role": "system", "content": "You are a judge."},
        {"role": "user", "content": "héllo — score this"},
    ]
    krepis_hash = krepis_content_hash(fixed_input_messages)
    nousergon_lib_hash = nousergon_lib_content_hash(fixed_input_messages)
    assert krepis_hash == nousergon_lib_hash, (
        f"content_hash algorithms diverged on a fixed input: "
        f"krepis={krepis_hash!r} nousergon_lib={nousergon_lib_hash!r} — "
        "both must canonicalize identically (sorted keys, ensure_ascii="
        "False, default=str) or live/replay duplicates of the same "
        "teacher call will NOT collapse under dedup."
    )

    # ... and the same must hold end-to-end for a full built record's
    # stored provenance.content_hash, not just the standalone helper.
    record = build_sft_record(_synthetic_llm_result(), producer="p")
    assert record["provenance"]["content_hash"] == nousergon_lib_content_hash(
        record["input_messages"]
    )
