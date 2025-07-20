import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Load FinBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Map sentiment to action
label_map = {
    "positive": ("BUY", "🟢"),
    "neutral": ("HOLD", "🟡"),
    "negative": ("SELL", "🔴")
}

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
        for company in companies:
            stock_recommendations[company] = (sentiment, action, symbol)

        print(f"{idx:02d}. {headline}\n   ➤ Sentiment: {sentiment.capitalize()} → {action} {symbol}\n")

    if stock_recommendations:
        print("\n📊 Summary of Recommended Actions:")
        for company, (sentiment, action, symbol) in stock_recommendations.items():
            print(f" - {company}: {action} {symbol}")
    else:
        print("\n⚠️ No company names detected.")

if __name__ == "__main__":
    main()
