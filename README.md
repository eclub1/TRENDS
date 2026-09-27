# ETH Trend AI

An AI-powered terminal tool that analyzes Ethereum market trends and detects when a trend change is likely.

## How it works

The engine pulls live ETH OHLC data from CoinGecko (free, no API key needed) and runs it through a multi-signal scoring system:

| Indicator | What it measures |
|---|---|
| EMA Alignment (9/21/50/200) | Structural trend direction |
| MACD | Momentum and crossovers |
| RSI | Overbought / oversold conditions |
| Bollinger Bands | Volatility squeeze and breakouts |
| Stochastic Oscillator | Short-term momentum peaks/troughs |
| Rate of Change (10-period) | Raw price momentum |

Each signal contributes a bullish (+) or bearish (−) score. The aggregate score maps to:

```
STRONG_UPTREND  → score ≥ +4
UPTREND         → score ≥ +1.5
NEUTRAL         → score between −1.5 and +1.5
DOWNTREND       → score ≤ −1.5
STRONG_DOWNTREND → score ≤ −4
```

### Reversal detection

On top of trend scoring, the AI separately watches for:
- RSI divergence (price and momentum moving opposite directions)
- MACD zero-line crossings
- EMA 9/21 crossover imminence
- RSI exiting overbought/oversold zones
- Bollinger Band squeeze (low volatility coiling for breakout)
- Stochastic extremes

Reversal risk is rated: **LOW → MEDIUM → HIGH → CRITICAL**

## Setup

```bash
cd eth_trend_ai
pip install -r requirements.txt
```

## Usage

```bash
# Single analysis (default: 90 days of data)
python main.py

# Use more historical data for better indicator accuracy
python main.py --days 180

# Auto-refresh every 5 minutes
python main.py --watch 300

# Plain text output (useful for scripting)
python main.py --raw
```

## Output

```
ETH Trend AI  ·  2026-09-27  01:25:00

┌──────────────────────────────────────────────────────┐
│  ETH / USD   $2,450.00   24h: +3.21%   7d: -1.40%   │
└──────────────────────────────────────────────────────┘

TREND ANALYSIS
┌─────────────────────────────────────────────────────┐
│  Trend               Reversal Risk     Confidence   │
│  📈 UPTREND          MEDIUM            67%          │
└─────────────────────────────────────────────────────┘
Score: ████████░░  +2.50

┌─── Signal Breakdown ─────────────────────────────────┐
│ EMA Alignment  │  2    │ +1.6 │ Mostly bullish ...   │
│ MACD           │  12.3 │ +1.5 │ MACD bullish ...     │
│ RSI            │  58.2 │ +0.8 │ RSI strong ...       │
│ ...            │       │      │                      │
└──────────────────────────────────────────────────────┘

⚠  MACD near zero-line crossing
⚠  EMA 9/21 crossover imminent

┌─── AI Recommendation ────────────────────────────────┐
│  Uptrend showing fatigue. Consider tightening stops  │
│  or partial profit-taking.                           │
└──────────────────────────────────────────────────────┘
```

## Disclaimer

This tool is for informational and educational purposes only. Nothing here constitutes financial advice. Always do your own research.
