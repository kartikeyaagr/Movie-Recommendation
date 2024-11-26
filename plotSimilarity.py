import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords


# Load the dataset
try:
    df = pd.read_csv("MOVIE_DATA_CHUNK_1.CSV")
except FileNotFoundError:
    print("The file MOVIE_DATA_CHUNK_1.CSV was not found.")
    exit()

# Split the dataset into train and test sets (80-20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


# Apply preprocessing to the 'Plot' column
train_df["cleaned_plot"] = train_df["Plot"].apply(preprocess_text)
test_df["cleaned_plot"] = test_df["Plot"].apply(preprocess_text)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_df["cleaned_plot"])
test_tfidf = vectorizer.transform(test_df["cleaned_plot"])


# Function to find similar movies
def find_similar_movies(test_plot_index, top_n=5):
    test_plot_vector = test_tfidf[test_plot_index]
    cosine_similarities = cosine_similarity(test_plot_vector, train_tfidf).flatten()
    similar_movie_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return train_df.iloc[similar_movie_indices]


# Example usage: Find similar movies for the first movie in the test set
similar_movies = find_similar_movies(0)
print(similar_movies[["Title", "Plot"]])
similar_movies.to_csv("similar_movies.csv", index=False)
