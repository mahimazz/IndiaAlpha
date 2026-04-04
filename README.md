# IndiaAlpha 🇮🇳📈

**Multi-agent AI system that analyses NSE/BSE Indian stocks using the investment philosophy of legendary investors — Warren Buffett, Rakesh Jhunjhunwala, Charlie Munger and more.**

---

## What is IndiaAlpha?

IndiaAlpha is an AI-powered stock analysis platform built for **Indian markets**. It runs multiple AI agents, each modelled on a real legendary investor, to independently analyse NSE/BSE stocks and generate a final buy/sell/hold decision with confidence scores.

Unlike the original open-source system (which only supports US stocks), IndiaAlpha has been fully extended to support **Nifty 50 and BSE stocks** using live market data — making it the first multi-agent hedge fund AI built specifically for Indian markets.

---

## Demo

> Run `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS` or any NSE/BSE ticker.

```
python3 src/main.py --ticker RELIANCE.NS --show-reasoning
```

**Sample Output:**

```
==========  Rakesh Jhunjhunwala Agent  ==========
Signal: BULLISH | Confidence: 78%
Reasoning: Reliance shows strong revenue CAGR with improving free 
cash flow. The conglomerate's diversification into Jio and retail 
aligns with India's consumption growth story...

==========  Warren Buffett Agent  ==========
Signal: BULLISH | Confidence: 82%
Reasoning: Strong moat via Jio's telecom dominance and retail 
network. Consistent earnings with improving ROE...

TRADING DECISION: BUY | Quantity: 12 | Confidence: 82%
```

---

## Key Features

- **17 AI Investor Agents** — Warren Buffett, Rakesh Jhunjhunwala, Charlie Munger, Peter Lynch, Michael Burry, Cathie Wood, Bill Ackman and more
- **Indian Market Support** — Full NSE (`RELIANCE.NS`) and BSE (`RELIANCE.BO`) ticker support via yfinance
- **Free LLM Integration** — Runs on Groq's free Llama 3.3 70B model (no OpenAI costs)
- **Volatility-Adjusted Risk Management** — Position sizing based on real volatility metrics
- **Web UI** — Full React frontend + FastAPI backend included
- **Zero Cost** — Free LLM (Groq) + free market data (yfinance)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Agents | LangGraph, LangChain |
| LLM | Groq (Llama 3.3 70B) — free |
| Indian Market Data | yfinance (NSE/BSE) |
| Backend | FastAPI, Python 3.13 |
| Frontend | React, TypeScript |
| Caching | SQLite |

---

## Supported Indian Stocks

Any NSE or BSE stock works — just append `.NS` for NSE or `.BO` for BSE:

| Company | Ticker |
|---------|--------|
| Reliance Industries | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| Infosys | `INFY.NS` |
| Wipro | `WIPRO.NS` |
| ITC | `ITC.NS` |
| Bajaj Finance | `BAJFINANCE.NS` |
| Asian Paints | `ASIANPAINT.NS` |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/mahimazz/IndiaAlpha.git
cd IndiaAlpha
```

### 2. Install dependencies

```bash
pip3 install -e .
pip3 install yfinance
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your free Groq API key (get one at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run analysis

```bash
python3 src/main.py --ticker RELIANCE.NS --show-reasoning
```

Select your analysts, choose **Llama 3.3 70B (Groq)** as the model, and watch the agents analyse the stock.

---

## What I Built On Top of the Base Project

This project extends the original [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) with significant modifications:

| What I Added | Details |
|-------------|---------|
| **Indian market data pipeline** | Built `src/utils/indian_stocks.py` — new module for NSE/BSE data via yfinance |
| **yfinance integration in api.py** | Added `_is_indian_ticker()`, `_get_yfinance_prices()`, `_get_yfinance_metrics()` functions |
| **Fixed all 17 investor agents** | Each agent rewritten to handle Indian financial data format and missing fields gracefully |
| **Groq free LLM support** | Added Llama 3.3 70B and Llama 3.1 8B to model list — runs at zero cost |
| **Bug fixes** | Fixed news sentiment crash, growth agent minimum periods bug, Stanley Druckenmiller EBITDA bug |

---

## Project Structure

```
IndiaAlpha/
├── src/
│   ├── agents/          ← 17 investor agents (Warren Buffett, Rakesh Jhunjhunwala etc.)
│   ├── tools/
│   │   └── api.py       ← Modified: yfinance integration for Indian stocks
│   ├── utils/
│   │   └── indian_stocks.py  ← New: Indian market data helpers
│   └── main.py
├── app/
│   ├── frontend/        ← React web UI
│   └── backend/         ← FastAPI backend
└── .env.example
```

---

## Investor Agents

| Agent | Investment Style |
|-------|----------------|
| Rakesh Jhunjhunwala | Growth focus, India-first, long-term conviction |
| Warren Buffett | Value investing, economic moat, margin of safety |
| Charlie Munger | Quality businesses at fair prices |
| Peter Lynch | Growth at reasonable price (GARP) |
| Michael Burry | Deep value, contrarian bets |
| Cathie Wood | Disruptive innovation, high growth |
| Bill Ackman | Activist investing, concentrated positions |
| Aswath Damodaran | Story + numbers valuation |
| Mohnish Pabrai | Cloning great investors, low risk high uncertainty |
| Phil Fisher | Scuttlebutt research, qualitative analysis |
| Stanley Druckenmiller | Macro trends, momentum |
| Nassim Taleb | Risk, uncertainty, black swan protection |
| Ben Graham | Father of value investing, net-nets |

---

## License

Based on the open-source [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) project. Extended with Indian market support.

---

*Built for Indian investors. Powered by AI. Zero cost to run.*

