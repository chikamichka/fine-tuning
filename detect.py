import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob

# Load the fine-tuned model and tokenizer
model_path = "final_bert_model"  # Adjust path if needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Make sure the model is in eval mode
model.eval()

# Helper: Sentiment analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Helper: Awareness of negativity
def contains_negative_awareness(text):
    awareness_keywords = ["bad", "unhealthy", "harmful", "not good", "avoid", "hate"]
    return any(word in text.lower() for word in awareness_keywords)

# Known healthy habits
known_healthy_habits = [
    "working out in the morning", "exercise", "eating vegetables",
    "drinking water", "sleeping well"
]

def classify_text(text):
    # Quick check for known healthy habits
    if any(habit in text.lower() for habit in known_healthy_habits):
        print("✅ That seems like a healthy habit!")
        return

    # Encode input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    unhealthy_prob, healthy_prob = probabilities[0].tolist()

    sentiment = analyze_sentiment(text)
    negative_awareness = contains_negative_awareness(text)

    print(f"\n🔹 Raw logits: {logits.tolist()}")
    print(f"🔹 Probabilities: {probabilities.tolist()}")
    print(f"🔹 Sentiment Polarity: {sentiment}")

    # Interpret the result (enhanced logic)
    healthy_keywords = [
    "no screen time", "sleep early", "wake up early", "morning routine", "journaling",
    "meditation", "working out", "exercise", "early workout", "less sugar", "drink water",
    "balanced diet", "healthy food", "yoga", "mental health", "bedtime routine"
    ]

    is_healthy_worded = any(word in text.lower() for word in healthy_keywords)

    if healthy_prob > 0.8 and sentiment > 0.2:
        print("✅ That seems like a healthy habit!")
    elif is_healthy_worded and (sentiment > 0 or healthy_prob > 0.5):
        print("✅ That sounds healthy !")
    elif prediction == 1:
        print("✅ This is a healthy habit!")
    elif prediction == 0 and sentiment < 0:
        print("🚨 That's an unhealthy habit.")
    elif abs(healthy_prob - unhealthy_prob) < 0.2:
        print("🤔 Not sure, context unclear.")
    else:
        print("⚠️ Take care — this might not be healthy.")



# Loop
while True:
    habit = input("\nEnter a habit (or type 'exit' to quit): ").strip().lower()
    if habit == "exit":
        print("Goodbye!")
        break
    classify_text(habit)
