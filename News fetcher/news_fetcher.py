import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

# Load spaCy and FinBERT
nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Telegram Bot Config
BOT_TOKEN = '7625630100:AAH5BLFfE0Egm-EKcFakBTSdN5sgAGChX1E'
CHAT_ID = '5562463187'

# Sentiment to action map
label_map = {
    "positive": ("BUY", "🟢"),
    "neutral": ("HOLD", "🟡"),
    "negative": ("SELL", "🔴")
}

def send_telegram_alert(stock_name, reason):
    message = f"📈 BUY Signal Detected for {stock_name}!\nReason: {reason}"
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("✅ Telegram Alert Sent!")
    else:
        print("❌ Failed to send alert:", response.text)

def fetch_headlines():
    urls = {
        "ET": "https://economictimes.indiatimes.com/markets",
        "Moneycontrol": "https://www.moneycontrol.com/news/business/markets/",
        "LiveMint": "https://www.livemint.com/market"
    }

    headlines = []
    for source, url in urls.items():
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            tags = soup.find_all("a")
            for tag in tags:
                text = tag.get_text(strip=True)
                if 10 < len(text) < 200:
                    headlines.append(text)
        except Exception as e:
            print(f"Failed to fetch from {source}: {e}")

    return list(set(headlines))[:15]

def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    sentiment = model.config.id2label[label].lower()
    return sentiment

def extract_company_names(text):
    doc = nlp(text)
    companies = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
    return companies

def main():
    print("📈 Indian Stock News Sentiment Analyzer")
    headlines = fetch_headlines()

    print(f"\n📰 Analyzing {len(headlines)} headlines from ET, Moneycontrol & LiveMint...\n")
    stock_recommendations = {}

    for idx, headline in enumerate(headlines, 1):
        sentiment = analyze_sentiment(headline)
        action, symbol = label_map[sentiment]
        companies = extract_company_names(headline)

        print(f"{idx:02d}. {headline}\n   ➤ Sentiment: {sentiment.capitalize()} → {action} {symbol}")

        # Save only if company names are found
        for company in companies:
            stock_recommendations[company] = (sentiment, action, symbol)
            # Send Telegram alert for BUY
            if action == "BUY":
                send_telegram_alert(company, headline)

    if stock_recommendations:
        print("\n📊 Summary of Recommended Actions:")
        for company, (sentiment, action, symbol) in stock_recommendations.items():
            print(f" - {company}: {action} {symbol}")
    else:
        print("\n⚠️ No company names detected.")

if __name__ == "__main__":
    main()
