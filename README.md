# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using Python and Machine Learning.

The system recommends movies similar to a selected movie by analyzing its overview, genres, keywords, cast, and director.

## 📌 Project Overview

This project uses Natural Language Processing and Machine Learning techniques to calculate similarity between movies.

When a user selects a movie, the system finds movies with similar characteristics and displays the top 10 recommendations.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## 🧠 Machine Learning Techniques

### 1. Data Cleaning

Movie and credit datasets are combined and cleaned.

### 2. Feature Engineering

The following movie features are used:

- Movie overview
- Genres
- Keywords
- Top 3 cast members
- Director

These features are combined into a single `tags` field.

### 3. Text Vectorization

`CountVectorizer` converts the movie tags into numerical feature vectors.

### 4. Similarity Calculation

Cosine similarity is used to calculate how similar movies are to each other.

### 5. Recommendation

The system returns the top 10 movies with the highest similarity scores.

## 📊 Dataset

The project uses the TMDB 5000 Movie Dataset.

The dataset contains information about approximately 4,800 movies, including movie details, genres, keywords, cast, crew, ratings, and release dates.

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── venv/
│
├── app.py
├── data_analysis.py
├── recommender.py
├── cleaned_movies.csv
├── requirements.txt
└── README.md