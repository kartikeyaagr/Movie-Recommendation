import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.express as px
import os

path = "dataset/"
extension = ".csv"

files = [file for file in os.listdir(path) if file.endswith(extension)]
dfs = []
for file in files:
    df = pd.read_csv(os.path.join(path, file))
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True)

movies = df

# Convert each genre in 'Genres' column to a list
movies["Genre"] = movies["Genre"].astype(str)
movies["Genre"] = movies["Genre"].apply(lambda x: x.split(", "))

# Join the list back to a string for vectorization
movies["Genre"] = movies["Genre"].apply(lambda x: " ".join(x))

# Vectorize the categorical columns
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))
genre_matrix = vectorizer.fit_transform(movies["Genre"])
language_matrix = vectorizer.fit_transform(movies["Language"].astype(str))
director_matrix = vectorizer.fit_transform(movies["Director"].astype(str))
production_matrix = vectorizer.fit_transform(movies["Production"].astype(str))

# Combine all vectors and the numerical column
combined_matrix = pd.concat(
    [
        pd.DataFrame(genre_matrix.toarray()),
        pd.DataFrame(language_matrix.toarray()),
        pd.DataFrame(director_matrix.toarray()),
        pd.DataFrame(production_matrix.toarray()),
        movies["IMDBRating"],
    ],
    axis=1,
)

# Standardize the combined matrix
scaler = StandardScaler()
combined_matrix_scaled = scaler.fit_transform(combined_matrix)

# Apply KNN clustering
num_neighbors = 5  # You can change the number of neighbors
knn = NearestNeighbors(n_neighbors=num_neighbors)
knn.fit(combined_matrix_scaled)
distances, indices = knn.kneighbors(combined_matrix_scaled)

# Reduce dimensionality using PCA for visualization
pca = PCA(n_components=2)
combined_matrix_2d = pca.fit_transform(combined_matrix_scaled)

# Add PCA components to the DataFrame
movies["PCA1"] = combined_matrix_2d[:, 0]
movies["PCA2"] = combined_matrix_2d[:, 1]

# Plot the clusters
fig = px.scatter(
    movies,
    x="PCA1",
    y="PCA2",
    hover_data=["Title", "Genre", "Language", "Director", "Production", "IMDBRating"],
)
fig.update_layout(title="Movie Clusters", xaxis_title="PCA1", yaxis_title="PCA2")
fig.show()

# Save the vectors to an output file
output_df = pd.DataFrame(combined_matrix_scaled)
output_df.to_csv("movie_vectors.csv", index=False)
