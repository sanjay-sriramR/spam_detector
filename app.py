import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Train model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Streamlit UI
st.title("📧 Simple Spam Detector")
st.write(f"Model trained with accuracy: **{accuracy*100:.2f}%**")

user_input = st.text_area("Enter a message:")

if st.button("Check"):
    prediction = model.predict([user_input])[0]
    if prediction == "spam":
        st.error("🚨 This is likely SPAM!")
    else:
        st.success("✅ This looks safe (HAM).")
