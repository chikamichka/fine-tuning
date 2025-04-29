import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    pipeline
)
from tqdm import tqdm
import warnings
from collections import Counter


warnings.filterwarnings("ignore")

# Load models
print("Loading models...")

# 1. Health classification model
HEALTH_MODEL_PATH = "final_bert_model"
health_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
health_model = BertForSequenceClassification.from_pretrained(HEALTH_MODEL_PATH)
health_model.eval()

# 2. Sentiment analysis model with proper tokenizer
sentiment_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512
)

# Prediction functions
def predict_health(text):
    try:
        inputs = health_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = health_model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        return "healthy" if pred == 1 else "unhealthy"
    except:
        return "error"

def get_sentiment(text):
    try:
        result = sentiment_analyzer(text[:1000], truncation=True)[0]  # Truncate to 1000 chars for safety
        return result['label'].lower()
    except:
        return "neutral"  # Fallback to neutral if analysis fails

def combined_prediction(text):
    health_label = predict_health(text)
    if health_label == "error":
        return "error"

    sentiment = get_sentiment(text)

    # Decision matrix
    if health_label == "healthy":
        return "healthy" if sentiment in ["positive", "neutral"] else "unhealthy"
    else:
        return "unhealthy" if sentiment in ["positive", "neutral"] else "healthy"

# Process CSV file with batch processing
def process_csv(input_file, output_file, text_column="text", batch_size=32):
    df = pd.read_csv(input_file)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    print("Processing...")

    # Process in batches
    results = []
    texts = df[text_column].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i+batch_size]
        batch_results = []

        for text in batch:
            health = predict_health(text)
            sentiment = get_sentiment(text)
            final = combined_prediction(text) if health != "error" else "error"
            batch_results.append({
                'health_prediction': health,
                'sentiment': sentiment,
                'final_prediction': final
            })

        results.extend(batch_results)

    # Add results to dataframe
    result_df = pd.DataFrame(results)
    df = pd.concat([df, result_df], axis=1)

    # Calculate counts (excluding errors)
    valid_predictions = df[df['final_prediction'] != "error"]['final_prediction']
    counts = Counter(valid_predictions)
    healthy_count = counts.get("healthy", 0)
    unhealthy_count = counts.get("unhealthy", 0)
    error_count = len(df) - len(valid_predictions)

    # Add summary
    summary_df = pd.DataFrame({
        text_column: ["CLASSIFICATION SUMMARY"],
        'health_prediction': [""],
        'sentiment': [""],
        'final_prediction': [
            f"Healthy: {healthy_count} | Unhealthy: {unhealthy_count} | Errors: {error_count}"
        ]
    })
    df = pd.concat([df, summary_df], ignore_index=True)

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print final counts
    print("\n=== FINAL CLASSIFICATION COUNTS ===")
    print(f"Healthy: {healthy_count}")
    print(f"Unhealthy: {unhealthy_count}")
    print(f"Errors: {error_count}")
    print("="*32)

if __name__ == "__main__":
    input_csv = "tweets.csv"  # Change to your file
    output_csv = "output_with_counts2.csv"
    text_column = "text"     # Change if different

    process_csv(input_csv, output_csv, text_column)
