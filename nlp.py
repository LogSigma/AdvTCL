import pandas as pd
import numpy as np
from konlpy.tag import Twitter
from gensim.models import word2vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

twitter = Twitter()

train_data = pd.read_csv('data/ratings_train.txt', sep='\t')

train_data = train_data[:4000]
train_data = [['/'.join(t) for t in twitter.pos(row[2], norm=True, stem=True)] for row in train_data.itertuples()]

num_features = 300 # 문자 벡터 차원 수
num_workers = 4 # 병렬 처리 스레드 수
context = 10 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

model = word2vec.Word2Vec(train_data, 
                          workers=num_workers, 
                          size=num_features, 
                          window=context,
                          sample=downsampling)

# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
model.save(model_name)

vocab = list(model.wv.vocab)
xmatrix = model[vocab]

print(len(xmatrix))
print(xmatrix[0][:10])

def elbow(X):
    sse = []

    for i in range(1,11):
        km = KMeans(n_clusters=i, init='k-means++', algorithm='auto', random_state=0)
        km.fit(X)
        sse.append(km.inertia_)

    plt.plot(range(1,11), sse, marker='o')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.show()

elbow(xmatrix)

k = 7
model = KMeans(n_clusters=k, init='k-means++', algorithm='auto')
model.fit(xmatrix)
final_df = pd.DataFrame(model.predict(xmatrix))
final_df.columns=['predict']

# 결과 합치기
final_df['vocab'] = pd.DataFrame(vocab)


tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(xmatrix)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()
