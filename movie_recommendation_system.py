import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle
from gensim.models import Word2Vec

df_reviews_one_sentences = pd.read_csv(
    "./data/movie_review_one_sentence_2015-2021.csv", index_col=0
)
Tfidf_matrix = mmread("./models/tfidf_movie_review.mtx").tocsr()
with open("./models/tfidf.pickle", "rb") as f:
    Tfidf = pickle.load(f)


def getRecommendation(cosine_sim):
    # movie_idx와 전체 martix의 cosine 유사도에 해당하는 값과 index list로 반환
    simScore = list(enumerate(cosine_sim[-1]))
    # 유사도의 내림차순으로 정렬
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    # 가장 유사한 상위 10개 영화
    simScore = simScore[1:11]
    # 10개 영화의 index
    movieidx = [i[0] for i in simScore]
    recMovieList = df_reviews_one_sentences.iloc[movieidx]
    return recMovieList


# movie_idx = df_reviews_one_sentences[df_reviews_one_sentences["titles"] == "기생충 (PARASITE)"].index[
#     0
# ]
# print(df_reviews_one_sentences.iloc[movie_idx])

# # 특정 영화와 전체 영화의 cosine 유사도 구함
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)  # shape: (1, len(Tfidf_matrix))
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)

embedding_model = Word2Vec.load("./models/word2VecModel_2015-2021.model")
key_word = "토르"
sentence = [key_word] * 10
if key_word in embedding_model.wv.index_to_key:
    sim_word = embedding_model.wv.most_similar(key_word, topn=10)
    labels = []
    for label, _ in sim_word:
        labels.append(label)
    print(labels)
    # 가장 유사한 단어를 많이 저장 ex) [겨울 겨울 겨울 가을 가을 여름]
    for i, word in enumerate(labels):
        sentence += [word] * (9 - i)

# 유사한 단어를 조합한 문장
sentence = " ".join(sentence)
print(sentence)

# 키워드에 해당하는 영화 추천
sentence_vec = Tfidf.transform([sentence])
print(sentence_vec)
print(sentence_vec.shape)
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
print(cosine_sim.shape)
recommendation = getRecommendation(cosine_sim)
print(recommendation["titles"])
