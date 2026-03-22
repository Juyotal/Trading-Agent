# Trading Agent: IBKR + Claude AI

Automated intraday trading agent that connects Interactive Brokers to Claude AI via a Python pipeline. The AI analyzes live market data and returns structured trade decisions which are executed automatically.

**Target instruments**: NQ (E-mini NASDAQ 100), ES (E-mini S&P 500), XAUUSD (Gold Spot)

## Architecture

```
IB Gateway ←→ Data Feed → Indicators → Claude AI → Risk Manager → Executor → IB Gateway
```

1. **Data Feed** pulls live OHLCV bars from IB Gateway
2. **Indicator Engine** computes EMA, RSI, ATR, VWAP, Bollinger Bands
3. **Claude Analyst** analyzes the market snapshot and returns a structured JSON trade decision
4. **Risk Manager** validates the decision against position limits, drawdown, and confidence thresholds
5. **Trade Executor** places bracket orders (entry + stop loss + take profit) through IB

## Prerequisites

- **Python 3.11+**
- **Interactive Brokers account** (paper trading is free)
- **IB Gateway or TWS** installed and running ([download here](https://www.interactivebrokers.com/en/trading/ib-api.php))
- **Anthropic API key** for Claude

## Setup

```bash
# Clone and enter the project
cd trading-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Configuration

### IB Gateway Setup

1. Download and install [IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) for Mac
2. Log in with your IBKR paper trading credentials
3. Go to **Configure → Settings → API → Settings**:
   - Enable "Enable ActiveX and Socket Clients"
   - Set Socket port to `7497` (paper)
   - Uncheck "Read-Only API"

### Environment Variables (.env)

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
TRADING_MODE=paper
```

### Trading Configuration (config/settings.yaml)

Edit `config/settings.yaml` to adjust:
- Instruments and their exchanges
- Analysis parameters (bar size, indicator periods)
- Risk limits (max position size, daily loss limit, confidence threshold)
- Schedule (analysis interval, market hours)

### Strategy Prompts (config/prompts/)

Edit the markdown files in `config/prompts/` to customize:
- `base_system.md` -- Core analysis rules, output format, risk framework
- `nq_strategy.md` -- NQ-specific context and strategy bias
- `es_strategy.md` -- ES-specific context and strategy bias
- `gold_strategy.md` -- XAUUSD-specific context and strategy bias

## Running

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the agent (make sure IB Gateway is running first)
python -m src.main
```

The agent will:
1. Connect to IB Gateway
2. Qualify contracts for configured instruments
3. Run analysis cycles every 5 minutes (configurable)
4. Skip instruments when their market is closed
5. Log all decisions and executions to `logs/`

Stop with `Ctrl+C` for graceful shutdown.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Safety Features

- **Paper trading by default** -- live mode requires explicit confirmation
- **Bracket orders only** -- every trade has a stop loss and take profit
- **Daily drawdown limit** -- auto-halts if daily loss exceeds threshold
- **Confidence gating** -- only executes trades above the confidence threshold
- **Consecutive loss circuit breaker** -- halts after N losses in a row
- **Connection loss protection** -- no new trades if disconnected (existing stops remain)
- **Full audit trail** -- every decision and execution logged as structured JSON

## Project Structure

```
trading-agent/
├── config/
│   ├── settings.yaml              # Instruments, risk params, schedule
│   └── prompts/                   # Editable Claude strategy prompts
├── src/
│   ├── main.py                    # Entry point + orchestration loop
│   ├── broker/                    # IB connection, data feed, executor
│   ├── analysis/                  # Indicators, snapshots, Claude analyst
│   ├── risk/                      # Risk manager
│   ├── models/                    # Pydantic schemas
│   └── utils/                     # Logging, scheduling
├── tests/                         # Unit tests
├── requirements.txt
└── .env.example
```
