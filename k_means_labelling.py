import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import os
from textblob import TextBlob
import numpy as np

movies = pd.read_csv("/home/kartikeya.agrawal_ug25/Movie-Recommendation/movies.csv")


def clean_data(df):
    # Remove the id column
    df = df.drop(columns=["id"])
    # Remove movies not in English
    df = df[df["original_language"] == "en"]
    # Remove specified columns
    df = df.drop(columns=["homepage", "backdrop_path", "budget", "runtime", "revenue"])
    # Remove specified columns
    df = df.drop(columns=["poster_path", "status", "vote_count"])
    # Extract release year from release_date
    df["release_year"] = pd.to_datetime(df["release_date"]).dt.year
    # Drop the original release_date column
    df = df.drop(columns=["release_date"])
    # Drop specified columns
    df = df.drop(
        columns=["title", "imdb_id", "original_language", "production_countries"]
    )
    # Rename vote_average to IMDBRating
    df = df.rename(columns={"vote_average": "IMDBRating"})
    # Capitalise all column names
    df.columns = [col.upper() for col in df.columns]
    # Capitalize first letter of all column names
    df.columns = [col.capitalize() for col in df.columns]
    # Rename columns production_companies to production, and spoken_languages to languages
    df = df.rename(
        columns={"production_companies": "production", "spoken_languages": "languages"}
    )
    # Rename Production_companies column to Production
    df = df.rename(columns={"Production_companies": "Production"})
    # Rename Spoken_languages column to Languages
    df = df.rename(columns={"Spoken_languages": "Languages"})
    # Replace all missing values with an empty string
    df = df.fillna("")
    # Convert specified columns to list objects
    df["Genres"] = df["Genres"].apply(lambda x: x.split(", "))
    df["Production"] = df["Production"].apply(lambda x: x.split(", "))
    df["Languages"] = df["Languages"].apply(lambda x: x.split(", "))
    df["Keywords"] = df["Keywords"].apply(lambda x: x.split(", "))
    
    return df

movies_clean = clean_data(movies.copy())
movies = movies_clean.copy()  # Create a fresh copy to avoid SettingWithCopyWarning
movies = movies[movies["Original_title"] != "Return"]

# Vectorize the categorical columns with a limit on the number of features
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "), max_features=5000)
genre_matrix = vectorizer.fit_transform(movies["Genres"].apply(lambda x: " ".join(x)))
keywords_matrix = vectorizer.fit_transform(
    movies["Keywords"].apply(lambda x: " ".join(x))
)
languages_matrix = vectorizer.fit_transform(
    movies["Languages"].apply(lambda x: " ".join(x))
)
production_matrix = vectorizer.fit_transform(
    movies["Production"].apply(lambda x: " ".join(x))
)

# Vectorize the Tagline column using TF-IDF with a limit on the number of features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tagline_matrix = tfidf_vectorizer.fit_transform(movies["Tagline"])

# Combine all vectors and the numerical columns
combined_matrix = pd.concat(
    [
        pd.DataFrame(genre_matrix.toarray()),
        pd.DataFrame(keywords_matrix.toarray()),
        pd.DataFrame(languages_matrix.toarray()),
        pd.DataFrame(production_matrix.toarray()),
        pd.DataFrame(tagline_matrix.toarray()),
        movies[["Release_year", "Imdbrating"]],
    ],
    axis=1,
)

# Ensure all column names are strings
combined_matrix.columns = combined_matrix.columns.astype(str)

# Standardize the combined matrix
scaler = StandardScaler()
combined_matrix_scaled = scaler.fit_transform(combined_matrix)

from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Fix invalid entries in combined_matrix
combined_matrix.fillna(0, inplace=True)
combined_matrix.replace([np.inf, -np.inf], 0, inplace=True)

# Validate after fixing
print("Invalid values in combined_matrix after fixing:")
print("NaN count:", np.isnan(combined_matrix).sum().sum())
print("Inf count:", np.isinf(combined_matrix).sum().sum())

# Retry scaling
combined_matrix_scaled = scaler.fit_transform(combined_matrix)

# Check for NaNs and infinite values after scaling
if np.isnan(combined_matrix_scaled).any() or np.isinf(combined_matrix_scaled).any():
    raise ValueError("Scaling resulted in invalid values. Check input data.")

# Apply KMeans clustering
num_clusters = 5  # You can change the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# Slice the scaled matrix to match the movies DataFrame length
combined_matrix_scaled_sliced = combined_matrix_scaled[:len(movies)]

# Fit and predict clusters
cluster_labels = kmeans.fit_predict(combined_matrix_scaled_sliced)

# Use .loc to assign clusters to avoid SettingWithCopyWarning
movies.loc[:, 'Cluster'] = cluster_labels

# Save the vectorizations before PCA
combined_matrix_df = pd.DataFrame(combined_matrix_scaled_sliced)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.7, svd_solver="full")
combined_matrix_pca = pca.fit_transform(combined_matrix_scaled_sliced)

# Save the vectorizations after PCA
combined_matrix_pca_df = pd.DataFrame(combined_matrix_pca)
# Plot the clusters
movies.loc[:, 'PCA1'] = combined_matrix_pca[:, 0]
movies.loc[:, 'PCA2'] = combined_matrix_pca[:, 1]


movies.to_csv("/home/kartikeya.agrawal_ug25/Movie-Recommendation/new_movies.csv", index=False)