from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import anthropic
from pydantic import ValidationError

from src.models.schemas import MarketSnapshot, TradeAction, TradeDecision
from src.utils.logger import get_logger

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def validate_decision_economics(
    decision: TradeDecision,
    current_price: float,
    min_reward_risk: float = 0.5,
    max_entry_drift_pct: float = 2.0,
) -> tuple[bool, str]:
    """Reject decisions that violate basic trade economics.

    LLMs occasionally emit stops on the wrong side of entry, targets closer
    than stops, or entry prices unrelated to current price. Schema validation
    won't catch any of this. These checks do.
    """
    if decision.action == TradeAction.HOLD:
        return True, ""

    if decision.entry_price is None or decision.stop_loss is None or decision.take_profit is None:
        return False, "missing entry/stop/target"

    entry = decision.entry_price
    sl = decision.stop_loss
    tp = decision.take_profit

    drift_pct = abs(entry - current_price) / current_price * 100 if current_price > 0 else 0
    if drift_pct > max_entry_drift_pct:
        return False, f"entry {entry} drifts {drift_pct:.2f}% from current {current_price}"

    if decision.action == TradeAction.BUY:
        if sl >= entry:
            return False, f"long stop {sl} at/above entry {entry}"
        if tp <= entry:
            return False, f"long target {tp} at/below entry {entry}"
        risk = entry - sl
        reward = tp - entry
    else:
        if sl <= entry:
            return False, f"short stop {sl} at/below entry {entry}"
        if tp >= entry:
            return False, f"short target {tp} at/above entry {entry}"
        risk = sl - entry
        reward = entry - tp

    if risk <= 0:
        return False, "zero stop distance"

    rr = reward / risk
    if rr < min_reward_risk:
        return False, f"reward:risk {rr:.2f} below minimum {min_reward_risk}"

    return True, ""


class ClaudeAnalyst:
    """Sends market snapshots to Claude and parses structured trade decisions."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.logger = get_logger()

    def _load_prompt(self, filename: str) -> str:
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text().strip()
        self.logger.warning(f"Prompt file not found: {path}")
        return ""

    def _build_system_prompt(self, instrument: str) -> str:
        base = self._load_prompt("base_system.md")

        instrument_map = {
            "NQ": "nq_strategy.md",
            "MNQ": "nq_strategy.md",
            "ES": "es_strategy.md",
            "MES": "es_strategy.md",
            "XAUUSD": "gold_strategy.md",
            "XAU": "gold_strategy.md",
            "MGC": "gold_strategy.md",
        }
        instrument_file = instrument_map.get(instrument.upper(), "")
        instrument_context = self._load_prompt(instrument_file) if instrument_file else ""

        parts = [base]
        if instrument_context:
            parts.append(f"\n\n## Instrument-Specific Context\n\n{instrument_context}")
        return "\n".join(parts)

    def _serialize_snapshot(self, snapshot: MarketSnapshot) -> str:
        data = snapshot.model_dump(mode="json")
        data["bars"] = data["bars"][-10:]
        return json.dumps(data, indent=2, default=str)

    async def analyze(self, snapshot: MarketSnapshot) -> Optional[TradeDecision]:
        system_prompt = self._build_system_prompt(snapshot.instrument)
        user_message = (
            f"Analyze this market data and return a trade decision as JSON.\n\n"
            f"```json\n{self._serialize_snapshot(snapshot)}\n```"
        )

        for attempt in range(1, self.max_retries + 2):
            try:
                self.logger.info(
                    f"Sending analysis request to Claude for {snapshot.instrument} "
                    f"(attempt {attempt})"
                )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                raw_text = response.content[0].text
                decision = self._parse_response(raw_text, snapshot.instrument)

                if decision:
                    ok, reason = validate_decision_economics(decision, snapshot.current_price)
                    if not ok:
                        self.logger.warning(
                            f"Rejecting Claude decision for {snapshot.instrument}: {reason}"
                        )
                        return None
                    self.logger.info(
                        f"Claude decision for {snapshot.instrument}: "
                        f"{decision.action.value} (confidence={decision.confidence})"
                    )
                    return decision

            except anthropic.APIError as e:
                self.logger.error(f"Claude API error (attempt {attempt}): {e}")
                if attempt > self.max_retries:
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error in Claude analysis (attempt {attempt}): {e}")
                if attempt > self.max_retries:
                    return None

        return None

    def _parse_response(self, raw_text: str, instrument: str) -> Optional[TradeDecision]:
        try:
            json_str = raw_text
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                json_str = raw_text.split("```")[1].split("```")[0]

            # Try to find JSON object in the text
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = json_str[start:end]

            data = json.loads(json_str)

            if "instrument" not in data:
                data["instrument"] = instrument

            decision = TradeDecision(**data)
            return decision

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from Claude response: {e}")
            self.logger.debug(f"Raw response: {raw_text[:500]}")
            return None
        except ValidationError as e:
            self.logger.error(f"Claude response failed schema validation: {e}")
            return None
