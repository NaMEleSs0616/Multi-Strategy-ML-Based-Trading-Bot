"""Strategy bank package."""

__all__ = ["MarketBundle", "build_strategy_bank"]


def __getattr__(name: str):
    if name in __all__:
        from strategies.bank import MarketBundle, build_strategy_bank

        return {"MarketBundle": MarketBundle, "build_strategy_bank": build_strategy_bank}[name]
    raise AttributeError(name)
