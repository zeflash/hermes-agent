"""Historical aggregates retain their cost coverage, not today's catalog prices."""

import pytest

from agent.insights import InsightsEngine
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, has_known_pricing
from hermes_state import SessionDB


_ROUTE = {"model": "claude-sonnet-4-20250514", "billing_provider": "anthropic"}


@pytest.mark.parametrize("per_route", [False, True])
@pytest.mark.parametrize("status,amount", [(None, 0.0), ("unknown", 0.0), ("estimated", 0.123)])
def test_historical_costs_are_not_repriced(tmp_path, per_route, status, amount):
    # The route is priced NOW; its historical call boundaries and past catalog
    # availability cannot be recovered from a token aggregate.
    assert has_known_pricing(_ROUTE["model"], _ROUTE["billing_provider"])
    with SessionDB(db_path=tmp_path / "history.db") as db:
        db.update_token_counts(
            "history", input_tokens=400_000, api_call_count=2,
            estimated_cost_usd=amount, cost_status=status, absolute=not per_route, **_ROUTE,
        )
        if status is None:
            # Legacy/imported rows can predate cost statuses entirely.
            db._conn.execute("UPDATE sessions SET cost_status = NULL")
            db._conn.execute("UPDATE session_model_usage SET cost_status = NULL")
            db._conn.commit()
        engine = InsightsEngine(db)
        report = engine.generate()
        assert report["overview"]["estimated_cost"] == pytest.approx(amount)
        assert report["overview"]["unknown_cost_sessions"] == (status != "estimated")
        assert report["models"][0]["cost"] == pytest.approx(amount)
        assert report["models"][0]["cost_status"] == (status or "unknown")


@pytest.mark.parametrize("writer", ["direct", "queued", "absolute", "auxiliary"])
@pytest.mark.parametrize("unknown_first", [False, True])
@pytest.mark.parametrize("unknown_status", [None, "unknown"])
@pytest.mark.parametrize("separate_buckets", [False, True])
def test_unknown_usage_is_not_erased_by_estimated_usage(tmp_path, writer, unknown_first, unknown_status, separate_buckets):
    priced = estimate_usage_cost(
        _ROUTE["model"], CanonicalUsage(input_tokens=200_000), provider=_ROUTE["billing_provider"],
    )
    assert priced.status == "estimated"
    amount = float(priced.amount_usd)
    with SessionDB(db_path=tmp_path / "mixed.db") as db:
        if writer == "auxiliary":
            # A known main-loop cost must not hide missing auxiliary pricing.
            db.update_token_counts(
                "mixed", model="main-model", input_tokens=1, api_call_count=1,
                estimated_cost_usd=0, cost_status="included",
            )
        entries = [(unknown_status, None), ("estimated", amount)]
        if not unknown_first:
            entries.reverse()
        total = 0.0
        for i, (status, cost) in enumerate(entries, 1):
            total += cost or 0.0
            if writer == "auxiliary":
                db.record_auxiliary_usage(
                    "mixed", "compression" if separate_buckets and i == 2 else "vision",
                    input_tokens=200_000, estimated_cost_usd=cost, **_ROUTE,
                )
            else:
                update = db.queue_token_counts if writer == "queued" else db.update_token_counts
                update(
                    "mixed", input_tokens=200_000 * (i if writer == "absolute" else 1),
                    api_call_count=i if writer == "absolute" else 1,
                    estimated_cost_usd=total if writer == "absolute" else cost,
                    cost_status=status, absolute=writer == "absolute",
                    billing_mode="other" if separate_buckets and i == 2 else "", **_ROUTE,
                )
        assert db.flush_token_counts()
        engine = InsightsEngine(db)
        report = engine.generate()
        assert report["overview"]["estimated_cost"] == pytest.approx(amount)
        assert report["models"][0]["cost"] == pytest.approx(amount)
        assert report["models"][0]["cost_status"] == "unknown"
        if writer != "auxiliary":
            assert db.get_session("mixed")["cost_status"] == "unknown"
        assert report["overview"]["unknown_cost_sessions"] == 1
        assert report["overview"]["included_cost_sessions"] == 0
        assert "unknown" in engine.format_gateway(report).lower()
