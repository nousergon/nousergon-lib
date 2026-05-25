"""
Unit tests for ``alpha_engine_lib.cost``.

Locks down the price-table contract: per-model effective-date lookup,
ordering invariants, malformed-YAML hard-fail, pure cost math correctness,
and the ``recompute_cost`` overwrite-vs-readonly modes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from alpha_engine_lib.cost import (
    PriceCard,
    PriceCardLookupError,
    PriceTable,
    PriceTableLoadError,
    compute_cost,
    load_default_pricing,
    load_pricing,
    metadata_from_anthropic_message,
    recompute_cost,
)
from alpha_engine_lib.decision_capture import ModelMetadata


# ── PriceCard ─────────────────────────────────────────────────────────────


class TestPriceCard:
    def test_minimal(self):
        c = PriceCard(
            model_name="claude-haiku-4-5",
            effective_from=date(2026, 1, 1),
            input_per_1m=1.0,
            output_per_1m=5.0,
            cache_read_per_1m=0.1,
            cache_create_per_1m=1.25,
        )
        assert c.input_per_1m == 1.0

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            PriceCard(
                model_name="x",
                effective_from=date(2026, 1, 1),
                input_per_1m=-1.0,
                output_per_1m=0.0,
                cache_read_per_1m=0.0,
                cache_create_per_1m=0.0,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            PriceCard(
                model_name="x",
                effective_from=date(2026, 1, 1),
                input_per_1m=1.0,
                output_per_1m=1.0,
                cache_read_per_1m=0.0,
                cache_create_per_1m=0.0,
                undocumented="oops",
            )


# ── PriceTable ordering invariants ────────────────────────────────────────


def _card(model_name: str, year: int, month: int, day: int, *, in_p: float = 1.0) -> PriceCard:
    return PriceCard(
        model_name=model_name,
        effective_from=date(year, month, day),
        input_per_1m=in_p,
        output_per_1m=in_p * 5,
        cache_read_per_1m=in_p * 0.1,
        cache_create_per_1m=in_p * 1.25,
    )


class TestPriceTableValidation:
    def test_single_card_ok(self):
        t = PriceTable(cards=[_card("haiku", 2026, 1, 1)])
        assert len(t.cards) == 1

    def test_multiple_models_independent(self):
        t = PriceTable(cards=[
            _card("haiku", 2026, 1, 1),
            _card("sonnet", 2026, 1, 1),
        ])
        assert len(t.cards) == 2

    def test_two_cards_one_model_ascending_ok(self):
        t = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
        ])
        assert len(t.cards) == 2

    def test_unsorted_rejected(self):
        with pytest.raises(PriceTableLoadError, match="not.*sorted"):
            PriceTable(cards=[
                _card("haiku", 2026, 6, 1),
                _card("haiku", 2026, 1, 1),
            ])

    def test_duplicate_effective_from_rejected(self):
        with pytest.raises(PriceTableLoadError, match="duplicate effective_from"):
            PriceTable(cards=[
                _card("haiku", 2026, 1, 1, in_p=1.0),
                _card("haiku", 2026, 1, 1, in_p=2.0),
            ])


# ── PriceTable.get ────────────────────────────────────────────────────────


class TestPriceTableLookup:
    def setup_method(self):
        self.table = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
            _card("sonnet", 2026, 1, 1, in_p=3.0),
        ])

    def test_returns_active_card_for_query_date(self):
        c = self.table.get("haiku", date(2026, 3, 15))
        assert c.input_per_1m == 1.0  # still on January card

    def test_returns_later_card_after_effective_date(self):
        c = self.table.get("haiku", date(2026, 6, 1))
        assert c.input_per_1m == 0.5  # June card is now active

        c = self.table.get("haiku", date(2026, 8, 1))
        assert c.input_per_1m == 0.5

    def test_accepts_datetime(self):
        c = self.table.get(
            "haiku", datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc),
        )
        assert c.input_per_1m == 1.0

    def test_unknown_model_hard_fails(self):
        with pytest.raises(PriceCardLookupError, match="opus"):
            self.table.get("opus", date(2026, 3, 15))

    def test_query_before_first_effective_date_hard_fails(self):
        with pytest.raises(PriceCardLookupError):
            self.table.get("haiku", date(2025, 12, 31))


# ── compute_cost ──────────────────────────────────────────────────────────


class TestComputeCost:
    def test_pure_input_only(self):
        card = _card("haiku", 2026, 1, 1, in_p=2.0)
        # 1M input tokens × $2/M = $2.00 exactly.
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
        )
        assert cost == pytest.approx(2.0)

    def test_combined_token_classes(self):
        card = PriceCard(
            model_name="haiku",
            effective_from=date(2026, 1, 1),
            input_per_1m=1.0,
            output_per_1m=5.0,
            cache_read_per_1m=0.1,
            cache_create_per_1m=1.25,
        )
        # input 4000 × 1.0 / 1M = 0.004
        # output 1200 × 5.0 / 1M = 0.006
        # cache_read 2000 × 0.1 / 1M = 0.0002
        # cache_create 500 × 1.25 / 1M = 0.000625
        # total = 0.010825
        cost = compute_cost(
            input_tokens=4000,
            output_tokens=1200,
            cache_read_tokens=2000,
            cache_create_tokens=500,
            card=card,
        )
        assert cost == pytest.approx(0.010825)

    def test_zero_tokens_is_zero_cost(self):
        card = _card("haiku", 2026, 1, 1, in_p=10.0)
        cost = compute_cost(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_create_tokens=0,
            card=card,
        )
        assert cost == 0.0


# ── recompute_cost ────────────────────────────────────────────────────────


class TestRecomputeCost:
    def setup_method(self):
        self.table = PriceTable(cards=[
            _card("haiku", 2026, 1, 1, in_p=1.0),
            _card("haiku", 2026, 6, 1, in_p=0.5),
        ])

    def test_overwrites_cost_usd_by_default(self):
        m = ModelMetadata(
            model_name="haiku",
            input_tokens=1_000_000,
            output_tokens=0,
            cost_usd=0.0,
        )
        cost = recompute_cost(m, self.table, at=date(2026, 3, 15))
        assert cost == pytest.approx(1.0)
        assert m.cost_usd == pytest.approx(1.0)

    def test_readonly_does_not_mutate(self):
        m = ModelMetadata(
            model_name="haiku",
            input_tokens=1_000_000,
            cost_usd=99.99,
        )
        cost = recompute_cost(
            m, self.table, at=date(2026, 3, 15), overwrite=False,
        )
        assert cost == pytest.approx(1.0)
        assert m.cost_usd == 99.99  # unchanged

    def test_uses_historical_card_for_historical_at(self):
        # Same call repriced against January vs June rates returns
        # different cost — proves effective-date routing works.
        m = ModelMetadata(model_name="haiku", input_tokens=1_000_000)
        jan_cost = recompute_cost(
            m, self.table, at=date(2026, 3, 15), overwrite=False,
        )
        jun_cost = recompute_cost(
            m, self.table, at=date(2026, 8, 1), overwrite=False,
        )
        assert jan_cost == pytest.approx(1.0)
        assert jun_cost == pytest.approx(0.5)

    def test_unknown_model_hard_fails(self):
        m = ModelMetadata(model_name="opus", input_tokens=1000)
        with pytest.raises(PriceCardLookupError):
            recompute_cost(m, self.table, at=date(2026, 3, 15))

    def test_default_at_uses_now(self):
        # No `at=` → now(UTC). The June card should resolve since today is
        # past 2026-06-01 in this test environment, but this test is
        # date-of-running-dependent. We just verify the call succeeds and
        # picks SOME valid card without raising.
        m = ModelMetadata(model_name="haiku", input_tokens=1000)
        cost = recompute_cost(m, self.table)
        assert cost > 0


# ── load_pricing ──────────────────────────────────────────────────────────


class TestLoadPricing:
    def test_loads_valid_yaml(self, tmp_path):
        yaml_path = tmp_path / "model_pricing.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: claude-haiku-4-5\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
        )
        table = load_pricing(yaml_path)
        assert len(table.cards) == 1
        assert table.cards[0].model_name == "claude-haiku-4-5"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pricing(tmp_path / "no_such_file.yaml")

    def test_missing_cards_key_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("not_cards: []\n")
        with pytest.raises(PriceTableLoadError, match="cards"):
            load_pricing(yaml_path)

    def test_top_level_list_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("- model_name: x\n")
        with pytest.raises(PriceTableLoadError):
            load_pricing(yaml_path)

    def test_extra_card_field_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: claude-haiku-4-5\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
            "    undocumented: 999\n"
        )
        with pytest.raises(ValueError):
            load_pricing(yaml_path)

    def test_unsorted_cards_rejected(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: haiku\n"
            "    effective_from: 2026-06-01\n"
            "    input_per_1m: 0.5\n"
            "    output_per_1m: 2.5\n"
            "    cache_read_per_1m: 0.05\n"
            "    cache_create_per_1m: 0.625\n"
            "  - model_name: haiku\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
        )
        with pytest.raises(PriceTableLoadError, match="not.*sorted"):
            load_pricing(yaml_path)


# ── load_default_pricing (packaged YAML) ──────────────────────────────────


class TestLoadDefaultPricing:
    def test_returns_pricetable_with_known_models(self):
        table = load_default_pricing()
        # Packaged file ships cards for the three current frontier models.
        # The exact rates may evolve; we only assert the models are present
        # so this test doesn't break on every price update.
        names = {c.model_name for c in table.cards}
        assert "claude-haiku-4-5" in names
        assert "claude-sonnet-4-6" in names
        assert "claude-opus-4-7" in names

    def test_default_card_lookup_works(self):
        table = load_default_pricing()
        card = table.get("claude-sonnet-4-6", date(2026, 5, 25))
        # Sonnet 4.x input rate is $3/M; locked here as a smoke check
        # that the packaged YAML actually parsed.
        assert card.input_per_1m == pytest.approx(3.0)


# ── metadata_from_anthropic_message (SDK adapter) ─────────────────────────


class _FakeUsage:
    """Duck-typed stand-in for ``anthropic.types.Usage``."""

    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _FakeMessage:
    """Duck-typed stand-in for ``anthropic.types.Message``."""

    def __init__(self, *, model: str, usage: _FakeUsage):
        self.model = model
        self.usage = usage


class TestMetadataFromAnthropicMessage:
    def test_basic_no_cache(self):
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=850, output_tokens=2700),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.model_name == "claude-sonnet-4-6"
        assert m.input_tokens == 850
        assert m.output_tokens == 2700
        assert m.cache_read_tokens == 0
        assert m.cache_create_tokens == 0
        assert m.cost_usd == 0.0  # caller fills via recompute_cost

    def test_with_cache_fields_populated(self):
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(
                input_tokens=100,
                output_tokens=200,
                cache_read_input_tokens=1500,
                cache_creation_input_tokens=2000,
            ),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.cache_read_tokens == 1500
        assert m.cache_create_tokens == 2000

    def test_none_cache_fields_zero_default(self):
        # Anthropic SDK returns None on these when caching wasn't used —
        # adapter must zero-default, not propagate None into ModelMetadata
        # (which would fail ge=0 validation).
        msg = _FakeMessage(
            model="claude-haiku-4-5",
            usage=_FakeUsage(
                input_tokens=10,
                output_tokens=20,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
        )
        m = metadata_from_anthropic_message(msg)
        assert m.cache_read_tokens == 0
        assert m.cache_create_tokens == 0

    def test_model_name_override(self):
        msg = _FakeMessage(
            model="claude-sonnet-4-6-20260101",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        m = metadata_from_anthropic_message(msg, model_name="claude-sonnet-4-6")
        assert m.model_name == "claude-sonnet-4-6"

    def test_integrates_with_recompute_cost(self):
        # End-to-end: SDK message → ModelMetadata → recompute against
        # the packaged default rate card. Locks the seam morning-signal
        # (and any other raw-SDK consumer) will actually use.
        msg = _FakeMessage(
            model="claude-sonnet-4-6",
            usage=_FakeUsage(input_tokens=1_000_000, output_tokens=0),
        )
        m = metadata_from_anthropic_message(msg)
        cost = recompute_cost(m, load_default_pricing(), at=date(2026, 5, 25))
        # 1M Sonnet input tokens @ $3/M = $3.00.
        assert cost == pytest.approx(3.0)
        assert m.cost_usd == pytest.approx(3.0)

