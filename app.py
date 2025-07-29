# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="📚 Book Recommender", layout="centered")

# Title
st.title("📚 Book Recommendation System")
st.caption("Built with Collaborative Filtering using Cosine Similarity")

# Load data
@st.cache_data
def load_data():
    books = pd.read_csv("data/Books.csv")
    ratings = pd.read_csv("data/Ratings.csv")
    return books, ratings

books, ratings = load_data()

# Merge book titles into ratings
ratings_with_name = ratings.merge(books, on="ISBN")

# Filter active users and famous books
x = ratings_with_name.groupby("User-ID").count()["Book-Rating"] > 200
active_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name["User-ID"].isin(active_users)]

y = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(famous_books)]

# Pivot table
pt = final_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
pt.fillna(0, inplace=True)

# Compute similarity
similarity_score = cosine_similarity(pt)

# Recommendation function
def recommend(book_name):
    try:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(
            list(enumerate(similarity_score[index])),
            key=lambda x: x[1], reverse=True
        )[1:6]
        recommended_books = [pt.index[i[0]] for i in similar_items]
        return recommended_books
    except IndexError:
        return []

# UI
st.subheader("🔎 Select a Book")
book_list = pt.index.tolist()
selected_book = st.selectbox("Choose a book you like", book_list)

if st.button("Recommend"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend(selected_book)
        if recommendations:
            st.success(f"Top 5 recommendations for: **{selected_book}**")
            for book in recommendations:
                st.write("👉", book)
        else:
            st.error("Book not found or not enough data to recommend.")
