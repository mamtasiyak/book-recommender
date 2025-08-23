import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


books = pd.read_csv("books/Books.csv") 
# books['ISBN'] = books['ISBN'].astype(str)


# Load ratings
@st.cache_data
def load_data():
    ratings = pd.read_csv('books/Ratings.csv')
    ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
    ratings.columns = ['user_id', 'book_id', 'rating']
    ratings = ratings[ratings['rating'] > 0]
    return ratings

ratings = load_data()

# Surprise setup
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

# Popular fallback
popular_books = (
    ratings.groupby('book_id')
    .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
    .reset_index()
)

def fallback_recommendations(top_n=5):
    top_books = popular_books.sort_values(
        by=['rating_count', 'avg_rating'], ascending=False
    ).head(top_n)
    return top_books[['book_id', 'avg_rating', 'rating_count']]

def recommend_books(user_id, top_n=5):
    if int(user_id) not in ratings['user_id'].values:
        print(f"\n New user detected. Showing fallback recommendations.")
        fallback = fallback_recommendations(top_n)
        return [
            (
                row['book_id'],
                books.loc[books['ISBN'] == str(row['book_id']), 'Book-Title'].values[0]
                if not books.loc[books['ISBN'] == str(row['book_id']), 'Book-Title'].empty
                else "Unknown Title",
                row['avg_rating']
            )
            for _, row in fallback.iterrows()
        ]   

    all_books = ratings['book_id'].unique()
    rated_books = ratings[ratings['user_id'] == int(user_id)]['book_id']
    unseen_books = [book for book in all_books if book not in rated_books.values]

    predictions = [model.predict(str(user_id), book_id) for book_id in unseen_books]
    top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]

    return [
        (
            pred.iid,
            books.loc[books['ISBN'] == pred.iid, 'Book-Title'].values[0]
            if not books.loc[books['ISBN'] == pred.iid, 'Book-Title'].empty
            else "Unknown Title",
            round(pred.est, 2)
        )
        for pred in top_preds
    ]


# ---------------- UI Layout ------------------
st.title("Book Recommender")

user_input = st.text_input("Enter your User ID:", value="276725")

if st.button("Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter a valid User ID.")
    else:
        recs = recommend_books(user_input, top_n=5)
        st.subheader("Recommended Books:")
        for i, (book_id, title, rating) in enumerate(recs, 1):
            st.write(f"{i}. {title} (Book ID: {book_id}) — {rating}")
