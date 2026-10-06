import pandas as pd
import ast

# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

# Remove duplicate title column from credits
credits = credits.drop(columns=["title"])

# Merge datasets
df = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)

# Keep only useful columns
df = df[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "vote_average",
        "release_date"
    ]
]

# Remove rows without a movie title
df = df.dropna(subset=["title"])

# Fill missing overviews
df["overview"] = df["overview"].fillna("")

# Convert JSON-like text into lists
def convert_to_list(text):
    try:
        data = ast.literal_eval(text)
        return [item["name"] for item in data]
    except:
        return []

df["genres"] = df["genres"].apply(convert_to_list)
df["keywords"] = df["keywords"].apply(convert_to_list)
df["cast"] = df["cast"].apply(convert_to_list)

# Display results
print("Cleaned Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSample Movie:")
print(df[["title", "genres", "keywords", "cast"]].head(1).to_string())

# Save cleaned dataset
df.to_csv("cleaned_movies.csv", index=False)

print("\nCleaned dataset saved as cleaned_movies.csv")