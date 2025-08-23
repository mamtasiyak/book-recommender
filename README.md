# Book Recommendation System

A simple Book Recommender System built using **Streamlit** and **Surprise (SVD)** algorithm, trained on the [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). It suggests personalized books for users based on collaborative filtering.

---

## Features

- Collaborative Filtering using SVD from `scikit-surprise`
- Intelligent fallback with most popular books for new users
- Trained on Book Ratings data with over 1M+ entries
- Recommends top N books with predicted ratings and titles
- Interactive UI built with **Streamlit**

---

## Tech Stack

- Python 3.10
- Streamlit
- Pandas
- scikit-surprise (for SVD model)

---

## Dataset Used

- `Books.csv`: Metadata of books (ISBN, title, etc.)
- `Ratings.csv`: User book ratings

> Source: [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/book-recommender-system.git
cd book-recommender-system
