
import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_movies.csv")

    posters = pd.read_csv("poster_mapping.csv")

    df = df.merge(
        posters[["id", "poster_path"]],
        on="id",
        how="left"
    )

    def parse_list(text):
        try:
            return ast.literal_eval(text)
        except:
            return []

    df["genres"] = df["genres"].apply(parse_list)
    df["keywords"] = df["keywords"].apply(parse_list)
    df["cast"] = df["cast"].apply(parse_list)

    return df


# --------------------------------------------------
# Create Recommendation Model
# --------------------------------------------------

@st.cache_resource
def create_model(df):

    def get_director(text):

        try:

            crew = ast.literal_eval(text)

            for person in crew:

                if person.get("job") == "Director":
                    return person.get("name", "")

            return ""

        except:
            return ""

    # Extract director
    df["director"] = df["crew"].apply(get_director)

    # Keep top 3 cast members
    df["cast"] = df["cast"].apply(
        lambda x: x[:3]
    )

    # Convert genres to lowercase
    df["genres"] = df["genres"].apply(
        lambda x: [
            item.lower().replace(" ", "")
            for item in x
        ]
    )

    # Convert keywords to lowercase
    df["keywords"] = df["keywords"].apply(
        lambda x: [
            item.lower().replace(" ", "")
            for item in x
        ]
    )

    # Convert cast to lowercase
    df["cast"] = df["cast"].apply(
        lambda x: [
            item.lower().replace(" ", "")
            for item in x
        ]
    )

    # Convert director to lowercase
    df["director"] = df["director"].apply(
        lambda x: x.lower().replace(" ", "")
    )

    # Convert overview to lowercase
    df["overview"] = df["overview"].fillna("").apply(
        lambda x: x.lower()
    )

    # Combine all features
    df["tags"] = (
        df["overview"]
        + " "
        + df["genres"].apply(
            lambda x: " ".join(x)
        )
        + " "
        + df["keywords"].apply(
            lambda x: " ".join(x)
        )
        + " "
        + df["cast"].apply(
            lambda x: " ".join(x)
        )
        + " "
        + df["director"]
    )

    # Count Vectorization
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        df["tags"]
    )

    # Cosine Similarity
    similarity = cosine_similarity(vectors)

    # Movie title index
    movie_indices = pd.Series(
        df.index,
        index=df["title"].str.lower()
    ).drop_duplicates()

    return similarity, movie_indices


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

def recommend_movies(
    movie_title,
    df,
    similarity,
    movie_indices
):

    movie_title = movie_title.lower()

    if movie_title not in movie_indices:
        return []

    index = movie_indices[movie_title]

    similarity_scores = list(
        enumerate(
            similarity[index]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in similarity_scores[1:11]:

        recommendations.append(
            {
                "id": df.iloc[i]["id"],

                "title": df.iloc[i]["title"],

                "score": score,

                "genres": df.iloc[i]["genres"],

                "vote_average": df.iloc[i].get(
                    "vote_average",
                    0
                ),

                "release_date": df.iloc[i].get(
                    "release_date",
                    ""
                ),

                "poster_path": df.iloc[i].get(
                    "poster_path",
                    None
                )
            }
        )

    return recommendations


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = load_data()


# --------------------------------------------------
# Create Model
# --------------------------------------------------

similarity, movie_indices = create_model(df)


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title(
    "🎬 Movie Recommendation System"
)

st.write(
    "Discover movies similar to your favorite "
    "films using content-based machine learning."
)

st.divider()


# --------------------------------------------------
# Movie Selection
# --------------------------------------------------

movie_list = sorted(
    df["title"].dropna().unique()
)

selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_list
)


# --------------------------------------------------
# Recommendation Button
# --------------------------------------------------

if st.button(
    "🔍 Recommend Movies",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_movie,
        df,
        similarity,
        movie_indices
    )

    if recommendations:

        st.subheader(
            f"🎯 Movies similar to **{selected_movie}**"
        )

        st.write(
            "Recommendations are based on movie "
            "overview, genres, keywords, cast, "
            "and director."
        )

        st.divider()

        # --------------------------------------------------
        # Display Recommendations
        # --------------------------------------------------

        for rank, movie in enumerate(
            recommendations,
            start=1
        ):

            col1, col2, col3 = st.columns(
                [1, 2, 7]
            )

            # ------------------------------
            # Rank
            # ------------------------------

            with col1:

                st.markdown(
                    f"## #{rank}"
                )

                st.metric(
                    "Similarity",
                    f"{movie['score'] * 100:.1f}%"
                )

            # ------------------------------
            # Poster
            # ------------------------------

            with col2:

                poster_url = movie["poster_path"]

                if pd.notna(poster_url) and poster_url:

                    st.image(
                        poster_url,
                        width=150
                    )

                else:

                    st.markdown(
                        "### 🎬"
                    )

                    st.write(
                        "Poster not available"
                    )

            # ------------------------------
            # Movie Details
            # ------------------------------

            with col3:

                st.markdown(
                    f"### 🎬 {movie['title']}"
                )

                # Genres
                genres = movie["genres"]

                if genres:

                    genre_text = ", ".join(
                        genre.title()
                        for genre in genres
                    )

                else:

                    genre_text = "Not available"

                st.write(
                    f"**Genres:** {genre_text}"
                )

                # Rating
                st.write(
                    f"⭐ **Rating:** "
                    f"{movie['vote_average']:.1f}/10"
                )

                # Release date
                if movie["release_date"]:

                    st.write(
                        f"📅 **Release Date:** "
                        f"{movie['release_date']}"
                    )

            st.divider()

    else:

        st.error(
            "Movie not found. Please select "
            "another movie."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Built with Python, Pandas, Scikit-learn "
    "and Streamlit"
)