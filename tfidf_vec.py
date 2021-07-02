import pickle

import pandas as pd
from scipy.io import mmwrite  # matrix 저장
from sklearn.feature_extraction.text import TfidfVectorizer

df_reviews_one_sentences = pd.read_csv(
    "./data/movie_review_one_sentence_2015_2021.csv", index_col=0
)
print(df_reviews_one_sentences.info())

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews_one_sentences["reviews"])

# 재사용하기위해 TfidfVectorizer 저장
with open("./models/tfidf.pickle", "wb") as f:
    pickle.dump(Tfidf, f)

# matrix 저장
mmwrite("./models/tfidf_movie_review.mtx", Tfidf_matrix)
