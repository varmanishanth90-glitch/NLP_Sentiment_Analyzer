import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

# Ensure NLTK resources are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Download punkt once at startup
nltk.download('punkt', quiet=True)

ps = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")  # columns: review, sentiment
data['cleaned'] = data['review'].apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned'])
y = data['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

# Landing page
@app.route("/", methods=["GET"])
def landing():
    return render_template("home.html")

# Sentiment Analyzer
@app.route("/index", methods=["GET", "POST"])
def sentiment_analyzer():
    if request.method == "POST":
        review = request.form.get("review", "")
        if not review.strip():
            return render_template("index.html", sentiment="⚠️ Please enter text")
        cleaned = preprocess(review)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            sentiment = "Positive 😊"
        elif prediction == 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        return render_template("index.html", sentiment=sentiment)
    return render_template("index.html")

# Text Summarizer
def summarize(text, num_sentences=3):
    from nltk.tokenize import sent_tokenize, word_tokenize
    words = word_tokenize(text.lower())
    freq = {}
    for word in words:
        if word.isalpha() and word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    sentences = sent_tokenize(text)
    scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                scores[sent] = scores.get(sent, 0) + freq[word]

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ' '.join(ranked[:num_sentences]) if ranked else "⚠️ No summary generated"

@app.route("/summarize", methods=["GET", "POST"])
def summarize_text():
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text.strip():
            return render_template("summarize.html", summary="⚠️ Please enter text")
        summary = summarize(text)
        return render_template("summarize.html", summary=summary)
    return render_template("summarize.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
