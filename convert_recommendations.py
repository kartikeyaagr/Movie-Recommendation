import pandas as pd
import json

# Load the uploaded files
movies_file_path = "movies.csv"
recommendations_file_path = "top_100_recommendations.csv"

# Read the CSV files into DataFrames
movies_df = pd.read_csv(movies_file_path)
recommendations_df = pd.read_csv(recommendations_file_path)

# Rename columns for better understanding
recommendations_df.columns = [
    f"Recommendation_{i+1}" for i in range(recommendations_df.shape[1])
]


# Function to fetch movie details based on IDs in recommendations
def get_movie_details(movie_ids):
    return movies_df[movies_df["id"].isin(movie_ids)]


# Preparing the data for each movie's recommendations
top_recommendations_details = {}
for idx, row in recommendations_df.iterrows():
    movie_ids = row.values.astype(
        float
    )  # Convert to float for comparison with 'id' in movies_df
    movie_details = get_movie_details(movie_ids)
    original_movie = movies_df.iloc[idx].to_dict()  # Get the original movie details
    top_recommendations_details[f"Movie_{idx+1}"] = {
        "Original_Movie": original_movie,
        "Recommendations": movie_details.to_dict(orient="records"),
    }

# Saving the output to a JSON file
output_file_path = "top_100_recommendations_details.json"
with open(output_file_path, "w") as json_file:
    json.dump(top_recommendations_details, json_file, indent=4)

print(f"Movie recommendations details saved to {output_file_path}")
