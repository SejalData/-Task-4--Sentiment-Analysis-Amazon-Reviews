import pandas as pd
from nrclex import NRCLex
from textblob import TextBlob

# Load CSV
df = pd.read_csv(
    r"C:\Users\sejal\OneDrive\Desktop\Task 4\Amazon_Reviews.csv",
    engine="python",
    on_bad_lines="skip"
)

# Remove missing reviews
df = df.dropna(subset=['Review Text'])

# Sentiment function
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['Review Text'].apply(get_sentiment)

# Emotion detection
def detect_emotion(text):
    emotion = NRCLex(text)
    return emotion.top_emotions

df['emotions'] = df['Review Text'].apply(detect_emotion)

# Show result
print(df[['Review Text', 'sentiment', 'emotions']].head())
df.to_csv("sentiment_output.csv", index=False)
