"""A catalog rate card belongs to a billing endpoint, not just a provider name."""

from decimal import Decimal

import pytest

from agent import models_dev, usage_pricing


@pytest.fixture
def catalog(monkeypatch):
    rates = {"input": 2, "output": 7, "cache_read": 0.25, "cache_write": 3}
    models = {name: {"cost": rates} for name in ("gpt-6-luna", "claude-sonnet-4-6")}
    monkeypatch.setattr(models_dev, "_models_dev_cache", {
        "opencode": {"models": models}, "opencode-go": {"models": models},
    })
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", lambda *a, **kw: {})


@pytest.mark.parametrize("provider,path", [
    ("opencode-zen", "/zen"), ("opencode-go", "/zen/go"),
])
@pytest.mark.parametrize("endpoint,known", [
    ("", True),
    ("https://opencode.ai{path}/v1", True),
    ("https://OPENCODE.AI:443{path}/v1/", True),
    ("https://opencode.ai{path}", True),  # Anthropic SDK appends /v1/messages.
    ("https://proxy.example{path}/v1", False),
    ("http://localhost:8000{path}/v1", False),
    ("http://opencode.ai{path}/v1", False),
    ("https://opencode.ai:8443{path}/v1", False),
    ("https://opencode.ai.example{path}/v1", False),
    ("https://other.opencode.ai{path}/v1", False),
    ("https://opencode.ai/other/v1", False),
    ("https://opencode.ai{path}/v1?route=other", False),
    ("https://opencode.ai{path}/v1#other", False),
    ("https://opencode.ai{path}/v1/extra", False),
])
def test_catalog_is_only_used_on_its_own_endpoint(catalog, provider, path, endpoint, known):
    base_url = endpoint.format(path=path)
    result = usage_pricing.estimate_usage_cost(
        "gpt-6-luna", usage_pricing.CanonicalUsage(input_tokens=1000),
        provider=provider, base_url=base_url,
    )
    assert result.status == ("estimated" if known else "unknown")
    assert result.amount_usd == (Decimal("0.002") if known else None)
    assert usage_pricing.has_known_pricing("gpt-6-luna", provider, base_url) is known
    other_path = "/zen/go" if path == "/zen" else "/zen"
    assert usage_pricing.get_pricing_entry(
        "gpt-6-luna", provider, f"https://opencode.ai{other_path}/v1",
    ) is None


@pytest.mark.parametrize("requested", ["opencode-zen", "zen", "opencode", "opencode_zen"])
@pytest.mark.parametrize("model", ["gpt-6-luna", "claude-sonnet-4-6"])
@pytest.mark.parametrize("base_url", [None, "https://proxy.example/zen/v1"])
def test_resolved_runtime_prices_only_the_selected_endpoint(catalog, monkeypatch, requested, model, base_url):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested=requested, target_model=model,
        explicit_api_key="test-key", explicit_base_url=base_url,
    )
    result = usage_pricing.estimate_usage_cost(
        model, usage_pricing.CanonicalUsage(input_tokens=1000),
        provider=runtime["provider"], base_url=runtime["base_url"],
    )
    assert result.status == ("unknown" if base_url else "estimated")

    # Custom endpoints may publish their own price; only the catalog fallback is refused.
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", lambda *a, **kw: {
        model: {"pricing": {"prompt": "0.000009", "completion": "0.000011"}},
    })
    entry = usage_pricing.get_pricing_entry(model, runtime["provider"], runtime["base_url"])
    assert entry.input_cost_per_million == Decimal(9)
    assert entry.pricing_version == "openai-compatible-models-api"
