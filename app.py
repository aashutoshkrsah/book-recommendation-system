# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
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

# Merge ratings with book info
ratings_with_name = ratings.merge(books, on="ISBN")

# Filter active users
active_users = ratings_with_name.groupby("User-ID").count()["Book-Rating"] > 200
filtered_users = active_users[active_users].index
filtered_rating = ratings_with_name[ratings_with_name["User-ID"].isin(filtered_users)]

# Filter popular books
famous_books = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
popular_books = famous_books[famous_books].index
final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(popular_books)]

# Pivot table for similarity computation
pt = final_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
pt.fillna(0, inplace=True)

# Cosine similarity matrix
similarity_score = cosine_similarity(pt)

# Recommendation function
def recommend(book_name):
    if book_name not in pt.index:
        return []
    
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_books = []
    for i in similar_items:
        title = pt.index[i[0]]
        book_info = books[books["Book-Title"] == title].drop_duplicates("Book-Title")
        if not book_info.empty:
            data = {
                "title": book_info["Book-Title"].values[0],
                "author": book_info["Book-Author"].values[0],
                "year": book_info["Year-Of-Publication"].values[0],
                "publisher": book_info["Publisher"].values[0],
                "image": book_info["Image-URL-M"].values[0]
            }
            recommended_books.append(data)
    return recommended_books

# UI - Book selection
st.subheader("🔎 Select a Book")
book_list = pt.index.tolist()

search = st.text_input("Search for a book:")
filtered_books = [book for book in book_list if search.lower() in book.lower()]
selected_book = st.selectbox("Choose a book you like", filtered_books if search else book_list)

# Recommendation display
if st.button("Recommend"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend(selected_book)
        if recommendations:
            st.success(f"Top 5 recommendations for: **{selected_book}**")
            for book in recommendations:
                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(book["image"], width=100)
                    with cols[1]:
                        st.markdown(f"**{book['title']}**")
                        st.markdown(f"*by {book['author']}*")
                        st.markdown(f"📅 {book['year']} &nbsp; | &nbsp; 🏢 {book['publisher']}")
                        st.markdown("---")
        else:
            st.error("No similar books found or insufficient data.")

# Footer
st.markdown("""<hr style="margin-top: 50px;">""", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        Made with ❤️ by <strong>Aashutosh Kumar Sah</strong>
    </div>
    """,
    unsafe_allow_html=True
)


