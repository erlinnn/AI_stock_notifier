#!/usr/bin/env python3
"""
AI Pre-Market Stock Intelligence System
A production-quality FinBERT-powered stock analysis engine
that sends high-confidence BUY alerts via Telegram.
"""

import os
import sys
import time
import logging
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("StockIntel")

# ─── Constants ─────────────────────────────────────────────
REQUEST_TIMEOUT = 15
MAX_HEADLINES = 30
BUY_THRESHOLD = 0.85
HOLD_LOW = 0.65
AGGREGATE_BUY_THRESHOLD = 0.80
BEARISH_RATIO = 0.50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

NSE_WHITELIST = {
    "reliance", "tcs", "infosys", "hdfc bank", "icici bank",
    "sbi", "state bank", "axis bank", "l&t", "larsen",
    "itc", "bharti airtel", "airtel", "hul", "hindustan unilever",
    "wipro", "tata motors", "maruti", "sun pharma",
    "ntpc", "power grid", "adani enterprises", "adani ports",
    "bajaj finance",
}

IGNORE_ENTITIES = {
    "india", "rbi", "reserve bank", "government", "market",
    "economy", "ministry", "sebi", "nse", "bse", "sensex",
    "nifty", "parliament", "lok sabha", "rajya sabha",
    "finance ministry", "modi", "pm",
}

CANONICAL_NAMES = {
    "reliance": "Reliance Industries",
    "tcs": "TCS",
    "infosys": "Infosys",
    "hdfc bank": "HDFC Bank",
    "icici bank": "ICICI Bank",
    "sbi": "SBI",
    "state bank": "SBI",
    "axis bank": "Axis Bank",
    "l&t": "L&T",
    "larsen": "L&T",
    "itc": "ITC",
    "bharti airtel": "Bharti Airtel",
    "airtel": "Bharti Airtel",
    "hul": "HUL",
    "hindustan unilever": "HUL",
    "wipro": "Wipro",
    "tata motors": "Tata Motors",
    "maruti": "Maruti Suzuki",
    "sun pharma": "Sun Pharma",
    "ntpc": "NTPC",
    "power grid": "Power Grid",
    "adani enterprises": "Adani Enterprises",
    "adani ports": "Adani Ports",
    "bajaj finance": "Bajaj Finance",
}


# ─── News Collection ──────────────────────────────────────
def fetch_economic_times() -> list[str]:
    """Fetch headlines from Economic Times Markets."""
    try:
        url = "https://economictimes.indiatimes.com/markets"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h2", "h3", "h4", "a"]):
            text = tag.get_text(strip=True)
            if len(text) > 25 and len(text) < 200:
                headlines.append(text)
        return headlines
    except Exception as e:
        log.warning(f"Economic Times fetch failed: {e}")
        return []


def fetch_moneycontrol() -> list[str]:
    """Fetch headlines from Moneycontrol Markets."""
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h2", "h3", "a"]):
            text = tag.get_text(strip=True)
            if len(text) > 25 and len(text) < 200:
                headlines.append(text)
        return headlines
    except Exception as e:
        log.warning(f"Moneycontrol fetch failed: {e}")
        return []


def fetch_livemint() -> list[str]:
    """Fetch headlines from LiveMint Markets."""
    try:
        url = "https://www.livemint.com/market"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h2", "h3", "a"]):
            text = tag.get_text(strip=True)
            if len(text) > 25 and len(text) < 200:
                headlines.append(text)
        return headlines
    except Exception as e:
        log.warning(f"LiveMint fetch failed: {e}")
        return []


def collect_headlines() -> list[str]:
    """Aggregate and deduplicate headlines from all sources."""
    log.info("Fetching headlines from 3 sources...")
    all_headlines = []
    all_headlines.extend(fetch_economic_times())
    all_headlines.extend(fetch_moneycontrol())
    all_headlines.extend(fetch_livemint())

    # Deduplicate (case-insensitive)
    seen = set()
    unique = []
    for h in all_headlines:
        key = h.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)

    headlines = unique[:MAX_HEADLINES]
    log.info(f"Collected {len(headlines)} unique headlines (from {len(all_headlines)} total)")
    return headlines


# ─── AI Sentiment Engine ──────────────────────────────────
class SentimentEngine:
    """FinBERT-powered financial sentiment analyser."""

    def __init__(self):
        log.info("Loading FinBERT model...")
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        log.info(f"FinBERT loaded on {self.device}")

    def analyse(self, text: str) -> dict:
        """Return sentiment label and confidence for a headline."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        scores = probs[0].cpu().tolist()
        label_scores = dict(zip(labels, scores))
        best_label = max(label_scores, key=label_scores.get)
        confidence = label_scores[best_label]

        # Decision
        if best_label == "positive" and confidence >= BUY_THRESHOLD:
            action = "BUY"
        elif best_label == "positive" and confidence >= HOLD_LOW:
            action = "HOLD"
        elif best_label == "negative":
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "sentiment": best_label,
            "confidence": round(confidence, 4),
            "action": action,
        }


# ─── Company Detection ────────────────────────────────────
class CompanyDetector:
    """spaCy + whitelist-based company extraction."""

    def __init__(self):
        log.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, headline: str) -> list[str]:
        """Return canonical company names found in headline."""
        doc = self.nlp(headline)
        found = set()

        # spaCy ORG entities
        for ent in doc.ents:
            if ent.label_ == "ORG":
                name_lower = ent.text.lower().strip()
                if name_lower in IGNORE_ENTITIES:
                    continue
                for key, canonical in CANONICAL_NAMES.items():
                    if key in name_lower or name_lower in key:
                        found.add(canonical)

        # Direct whitelist scan (catches abbreviations spaCy misses)
        headline_lower = headline.lower()
        for key, canonical in CANONICAL_NAMES.items():
            if key in headline_lower:
                found.add(canonical)

        # Remove if entity is in ignore list (double-check)
        found = {c for c in found if c.lower() not in IGNORE_ENTITIES}
        return list(found)


# ─── Aggregation & Alerting ───────────────────────────────
def aggregate_signals(
    headlines: list[str], engine: SentimentEngine, detector: CompanyDetector
) -> dict:
    """
    Analyse all headlines and aggregate per company.
    Returns dict of company -> aggregated signal info.
    """
    company_data = defaultdict(lambda: {
        "sentiments": [],
        "confidences": [],
        "headlines": [],
        "actions": [],
    })
    all_sentiments = []

    for headline in headlines:
        result = engine.analyse(headline)
        all_sentiments.append(result["sentiment"])
        companies = detector.extract(headline)

        for company in companies:
            data = company_data[company]
            data["sentiments"].append(result["sentiment"])
            data["confidences"].append(result["confidence"])
            data["headlines"].append(headline)
            data["actions"].append(result["action"])

    # Market mood
    neg_count = all_sentiments.count("negative")
    total = len(all_sentiments) if all_sentiments else 1
    bearish = (neg_count / total) >= BEARISH_RATIO
    market_mood = "Bearish" if bearish else ("Bullish" if neg_count / total < 0.3 else "Neutral")

    log.info(f"Market Mood: {market_mood} (negative ratio: {neg_count}/{total})")

    # Per-company aggregation
    signals = {}
    for company, data in company_data.items():
        pos_count = data["sentiments"].count("positive")
        neg_count_c = data["sentiments"].count("negative")
        total_c = len(data["sentiments"])
        avg_conf = sum(data["confidences"]) / total_c

        if pos_count > neg_count_c and avg_conf >= AGGREGATE_BUY_THRESHOLD:
            final_action = "BUY"
        elif neg_count_c > pos_count:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Bearish override
        if bearish and final_action == "BUY":
            final_action = "HOLD"
            log.info(f"  ⚠ {company}: BUY → HOLD (bearish market override)")

        # Best headline (highest confidence positive)
        best_headline = ""
        best_conf = 0
        for i, s in enumerate(data["sentiments"]):
            if s == "positive" and data["confidences"][i] > best_conf:
                best_conf = data["confidences"][i]
                best_headline = data["headlines"][i]

        signals[company] = {
            "action": final_action,
            "avg_confidence": round(avg_conf, 4),
            "positive_count": pos_count,
            "negative_count": neg_count_c,
            "news_count": total_c,
            "best_headline": best_headline or data["headlines"][0],
            "market_mood": market_mood,
        }

        log.info(
            f"  {company:25s} │ conf: {avg_conf:.2f} │ "
            f"+{pos_count}/-{neg_count_c} │ {final_action}"
        )

    return signals


def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not bot_token or not chat_id:
        log.error("BOT_TOKEN or CHAT_ID not set. Skipping Telegram.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        log.info("Telegram message sent ✓")
        return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


def format_buy_alert(company: str, info: dict) -> str:
    """Format a BUY signal Telegram message."""
    return (
        f"📈 <b>AI STOCK BUY SIGNAL</b>\n\n"
        f"<b>Company:</b> {company}\n"
        f"<b>Confidence:</b> {info['avg_confidence']:.0%}\n"
        f"<b>News Count:</b> {info['news_count']} headlines\n"
        f"<b>Positive Signals:</b> {info['positive_count']}\n"
        f"<b>Top Reason:</b> {info['best_headline']}\n"
        f"<b>Market Mood:</b> {info['market_mood']}\n"
    )


# ─── Main Pipeline ────────────────────────────────────────
def main():
    start = time.time()
    log.info("=" * 60)
    log.info("AI Pre-Market Stock Intelligence System")
    log.info("=" * 60)

    # 1. Collect headlines
    headlines = collect_headlines()
    if not headlines:
        log.warning("No headlines fetched. Sending fallback alert.")
        send_telegram("⚠️ No headlines could be fetched today. Check news sources.")
        return

    # 2. Load models
    engine = SentimentEngine()
    detector = CompanyDetector()

    # 3. Aggregate signals
    log.info("Analysing headlines...")
    signals = aggregate_signals(headlines, engine, detector)

    # 4. Send alerts (deduplicated)
    buy_signals = {k: v for k, v in signals.items() if v["action"] == "BUY"}
    sent = set()

    if buy_signals:
        for company, info in sorted(
            buy_signals.items(), key=lambda x: x[1]["avg_confidence"], reverse=True
        ):
            if company not in sent:
                send_telegram(format_buy_alert(company, info))
                sent.add(company)
    else:
        send_telegram(
            "📊 <b>Daily Market Summary</b>\n\n"
            "No strong BUY signals today.\n"
            "Market sentiment is cautious.\n"
            f"Headlines analysed: {len(headlines)}\n"
            f"Companies tracked: {len(signals)}"
        )

    elapsed = time.time() - start
    log.info(f"Pipeline complete in {elapsed:.1f}s — {len(sent)} BUY alerts sent")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
