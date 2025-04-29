import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import requests

# Load fine-tuned model and tokenizer
model_path = "final_bert_model"  # Adjust path as needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# OpenRouter API details
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Helpers
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def contains_negative_awareness(text):
    awareness_keywords = ["bad", "unhealthy", "harmful", "not good", "avoid", "hate"]
    return any(word in text.lower() for word in awareness_keywords)

def generate_persuasive_message(unhealthy_habit):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"Reply to this tweet persuasively, encouraging a healthier choice: {unhealthy_habit}"

    data = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "Respond like a tweet reply—concise, persuasive, and engaging."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    response = requests.post(OPENROUTER_URL, json=data, headers=headers)
    if response.status_code == 200:
        message = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
        print("🗣️ Persuasive Message:", message)
    else:
        print("❌ Error generating message:", response.text)

# Predefined lists
healthy_keywords = [
    "no screen time", "sleep early", "wake up early", "morning routine", "journaling",
    "meditation", "working out", "exercise", "early workout", "less sugar", "drink water",
    "balanced diet", "healthy food", "yoga", "mental health", "bedtime routine"
]

known_healthy_habits = [
    "working out in the morning", "exercise", "eating vegetables",
    "drinking water", "sleeping well"
]

# Classifier Logic
def classify_text(text):
    # Known healthy overrides
    if any(habit in text.lower() for habit in known_healthy_habits):
        print("✅ That seems like a healthy habit!")
        return

    # Model inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    unhealthy_prob, healthy_prob = probabilities[0].tolist()  # index 0: unhealthy, 1: healthy

    sentiment = analyze_sentiment(text)
    negative_awareness = contains_negative_awareness(text)
    is_healthy_worded = any(word in text.lower() for word in healthy_keywords)

    # Logs
    print(f"\n🔹 Raw logits: {logits.tolist()}")
    print(f"🔹 Probabilities: {probabilities.tolist()}")
    print(f"🔹 Sentiment Polarity: {sentiment}")

    # Final logic
    if healthy_prob > 0.8 and sentiment > 0.2:
        print("✅ That seems like a healthy habit!")
    elif is_healthy_worded and (sentiment > 0 or healthy_prob > 0.5):
        print("✅ That sounds healthy!")
    elif prediction == 1:
        print("✅ This is a healthy habit!")
    elif prediction == 0 and sentiment < 0:
        print("🚨 That's an unhealthy habit.")
        generate_persuasive_message(text)
    elif negative_awareness and prediction == 1:
        print("⚠️ You seem aware this is unhealthy!")
    elif abs(healthy_prob - unhealthy_prob) < 0.2:
        print("🤔 Not sure, context unclear.")
    else:
        print("⚠️ Take care — this might not be healthy.")
        generate_persuasive_message(text)

# Interactive loop
while True:
    habit = input("\nEnter a habit (or type 'exit' to quit): ").strip().lower()
    if habit == "exit":
        print("Goodbye!")
        break
    classify_text(habit)
