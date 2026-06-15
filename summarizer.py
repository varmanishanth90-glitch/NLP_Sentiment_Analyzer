import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def summarize(text, num_sentences=3):
    stop_words = set(stopwords.words('english'))
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
    return ' '.join(ranked[:num_sentences])
