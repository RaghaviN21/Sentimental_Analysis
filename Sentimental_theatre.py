import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
data = {
    "review": [
        "Amazing performance and great story",
        "Worst show ever",
        "Loved the acting",
        "Very boring and slow",
        "Fantastic direction and music",
        "Not worth watching"
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# 🔹 User input loop
while True:
    user_review = input("\nEnter your theatre review (or type 'exit' to quit): ")

    if user_review.lower() == "exit":
        print("Program ended 👋")
        break

    # Convert input
    user_vec = vectorizer.transform([user_review])

    # Predict
    prediction = model.predict(user_vec)

    # Output
    if prediction[0] == 1:
        print("👉 Sentiment: Positive 😊")
    else:
        print("👉 Sentiment: Negative 😡")