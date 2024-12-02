import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the post-PCA vectors
combined_matrix_pca_df = pd.read_csv(
    "/home/kartikeya.agrawal_ug25/Movie-Recommendation/movie_vectors_after_pca.csv"
)

# Load the original dataset
movies = pd.read_csv("/home/kartikeya.agrawal_ug25/Movie-Recommendation/movies.csv")

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(combined_matrix_pca_df)


# Function to get top 5 recommendations for each movie
def get_recommendations(similarity_matrix, top_n=5):
    recommendations = {}
    for idx in range(similarity_matrix.shape[0]):
        # Get the indices of the top_n most similar movies
        similar_indices = similarity_matrix[idx].argsort()[-top_n - 1 : -1][::-1]
        recommendations[idx] = similar_indices
    return recommendations


# Get recommendations for the top 100 rows
top_100_recommendations = get_recommendations(similarity_matrix[:100])

# Print recommendations for the first movie in the top 100
first_movie_recommendations = top_100_recommendations[0]
print("Recommendations for the first movie:")
for rec_idx in first_movie_recommendations:
    print(movies.iloc[rec_idx]["title"])

# Save the recommendations to a CSV file
recommendations_df = pd.DataFrame.from_dict(top_100_recommendations, orient="index")
recommendations_df.to_csv(
    "/home/kartikeya.agrawal_ug25/Movie-Recommendation/top_100_recommendations.csv",
    index=False,
)
