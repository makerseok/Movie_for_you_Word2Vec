from types import new_class
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = "C:/Windows/Fonts/NanumSquareL.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams["axes.unicode_minus"] = False
rc("font", family=font_name)

embedding_model = Word2Vec.load("./models/word2VecModel_2015-2021.model")
# print(embedding_model.wv.index_to_key)
print(len(embedding_model.wv))

# 키워드와 가장 유사한 10개 단어
key_word = "사랑"
sim_word = embedding_model.wv.most_similar(key_word, topn=10)
print(sim_word)

# 단어별 100개 차원 특성 데이터프레임으로 저장
vectors = []
labels = []
for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv.get_vector(label))

df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

# 차원 축소
tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(df_vectors)
df_xy = pd.DataFrame({"words": labels, "x": new_values[:, 0], "y": new_values[:, 1]})
print(df_xy.head())
print(df_xy.shape)

# 2차원으로 축소한 데이터 plot
df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)
plt.figure(figsize=(8, 8))
plt.scatter(0, 0, s=1500, marker="*")
for i in range(len(df_xy.x)):
    a = df_xy.loc[[i, 10], :]
    plt.plot(a.x, a.y, "-D", linewidth=2)
    plt.annotate(
        df_xy.words[i],
        xytext=(5, 2),
        xy=(df_xy.x[i], df_xy.y[i]),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )
plt.show()
