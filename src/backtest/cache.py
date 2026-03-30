from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from src.models.schemas import MarketSnapshot, TradeDecision


class ResponseCache:
    """File-backed cache for Claude analyst responses keyed on snapshot hash.

    Re-running a backtest against unchanged data and prompts costs nothing.
    """

    def __init__(self, cache_dir: str = ".backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def _key(snapshot: MarketSnapshot, prompt_fingerprint: str) -> str:
        payload = snapshot.model_dump_json()
        raw = f"{prompt_fingerprint}|{payload}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, snapshot: MarketSnapshot, prompt_fingerprint: str) -> Optional[TradeDecision]:
        key = self._key(snapshot, prompt_fingerprint)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return TradeDecision(**data)
        except Exception:
            return None

    def put(self, snapshot: MarketSnapshot, prompt_fingerprint: str, decision: TradeDecision):
        key = self._key(snapshot, prompt_fingerprint)
        path = self.cache_dir / f"{key}.json"
        path.write_text(decision.model_dump_json())
