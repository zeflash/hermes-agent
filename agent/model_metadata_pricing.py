"""Normalize token rates published by OpenAI-compatible model catalogs."""

from typing import Any


# An explicit unit wins. Without one, >= $0.001/token ($1,000/MTok) is a per-million quote.
_PRICING_UNIT_DIVISORS = {
    "per_token": 1, "per_1k_tokens": 1_000, "per_thousand_tokens": 1_000,
    "per_1m_tokens": 1_000_000, "per_million_tokens": 1_000_000,
}
_PER_MILLION_QUOTE_MIN = 0.001
_TOKEN_RATE_FIELDS = ("prompt", "completion", "cache_read", "cache_write")


def normalize_token_rates(pricing: dict[str, Any], unit: Any) -> dict[str, Any]:
    """Rescale the generic path's token rates to per-token strings (the contract usage_pricing
    multiplies by 1e6), the way the Novita/DeepInfra branches already do for their known units."""
    rates: dict[str, float] = {}
    for key in _TOKEN_RATE_FIELDS:
        try:
            rates[key] = float(pricing[key])
        except (KeyError, TypeError, ValueError):
            continue
    divisor = _PRICING_UNIT_DIVISORS.get(str(unit or "").strip().lower())
    if divisor is None:
        divisor = 1_000_000 if any(v >= _PER_MILLION_QUOTE_MIN for v in rates.values()) else 1
    if divisor != 1:
        pricing.update({key: str(value / divisor) for key, value in rates.items()})
    return pricing
