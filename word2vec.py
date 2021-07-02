import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv("./data/movie_review_2015-2021.csv")
print(review_word.info())

cleaned_token_reviews = list(review_word["cleaned_sentences"])

cleaned_tokens = []
count = 0
for sentence in cleaned_token_reviews:
    token = sentence.split(" ")
    cleaned_tokens.append(token)

embedding_model = Word2Vec(
    cleaned_tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1
)  # vector_size: 차원 수, windows: 앞 뒤로 고려하는 단어의 수, workers: 사용 cpu 수, min_count: 단어의 최소 등장 횟수
embedding_model.save("./models/word2VecModel_2015-2021.model")
print(embedding_model.wv.vocab.keys())
print(len(embedding_model.wv.vocab.keys()))
