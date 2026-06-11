import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")  # columns: review, sentiment

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

data['cleaned'] = data['review'].apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned'])
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        review = request.form["review"]
        cleaned = preprocess(review)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        return render_template("index.html", result=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    # IMPORTANT: Bind to PORT Render provides
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
