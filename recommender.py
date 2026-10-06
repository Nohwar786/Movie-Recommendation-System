import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load cleaned dataset
df = pd.read_csv("cleaned_movies.csv")


# Convert list-like columns back into Python lists
def parse_list(text):
    try:
        return ast.literal_eval(text)
    except:
        return []


df["genres"] = df["genres"].apply(parse_list)
df["keywords"] = df["keywords"].apply(parse_list)
df["cast"] = df["cast"].apply(parse_list)


# Extract director from crew
def get_director(text):
    try:
        crew = ast.literal_eval(text)

        for person in crew:
            if person.get("job") == "Director":
                return person.get("name", "")

        return ""
    except:
        return ""


df["director"] = df["crew"].apply(get_director)


# Keep only the first 3 cast members
df["cast"] = df["cast"].apply(lambda x: x[:3])


# Convert all features to lowercase
df["genres"] = df["genres"].apply(
    lambda x: [item.lower().replace(" ", "") for item in x]
)

df["keywords"] = df["keywords"].apply(
    lambda x: [item.lower().replace(" ", "") for item in x]
)

df["cast"] = df["cast"].apply(
    lambda x: [item.lower().replace(" ", "") for item in x]
)

df["director"] = df["director"].apply(
    lambda x: x.lower().replace(" ", "")
)


# Convert overview to lowercase words
df["overview"] = df["overview"].fillna("").apply(
    lambda x: x.lower()
)


# Create combined feature tags
df["tags"] = (
    df["overview"] + " "
    + df["genres"].apply(lambda x: " ".join(x)) + " "
    + df["keywords"].apply(lambda x: " ".join(x)) + " "
    + df["cast"].apply(lambda x: " ".join(x)) + " "
    + df["director"]
)


# Create Count Vectorizer
vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(df["tags"])


# Calculate cosine similarity
similarity = cosine_similarity(vectors)


print("Feature matrix shape:", vectors.shape)
print("Similarity matrix shape:", similarity.shape)

print("\nDirector of first movie:", df.iloc[0]["director"])
print("\nTags for first movie:")
print(df.iloc[0]["tags"][:500])
# Create movie index
movie_indices = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


# Recommendation function
def recommend_movies(movie_title, number_of_recommendations=10):

    movie_title = movie_title.lower()

    if movie_title not in movie_indices:
        print(f"\nMovie '{movie_title}' not found in the dataset.")
        return

    index = movie_indices[movie_title]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity[index])
    )

    # Sort movies by similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top recommendations
    similarity_scores = similarity_scores[1:number_of_recommendations + 1]

    print(f"\nMovies similar to '{df.iloc[index]['title']}':\n")

    for i, score in similarity_scores:
        print(
            f"{df.iloc[i]['title']} "
            f"(Similarity: {score:.2f})"
        )


# Test recommendation system
recommend_movies("Avatar", 10)
print("\n" + "=" * 50)

recommend_movies("The Dark Knight", 10)