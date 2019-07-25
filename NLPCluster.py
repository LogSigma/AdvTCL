# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import font_manager, rc
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

class NLPCluster:
	def __init__(self):
		self.N_CLUSTERS = 20
		self.N_COMPONENTS = 2
		self.FONT_DIR = "C:/Windows/Fonts/맑은 고딕.ttf"

	def search_k_means(self, X):
	"""
	최적 k 값 찾기
        """
        sse = []
		for i in range(1, self.N_CLUSTERS+1):
			km = KMeans(n_clusters=i, n_jobs=4, random_state=0)
			km.fit(X)
			sse.append(km.inertia_)

		plt.plot(range(1, self.N_CLUSTERS+1), sse, marker='o')


	def kmeans(self, k, X):
	"""
	kmeans++
        """
        kms = KMeans(n_clusters=k, n_jobs=4, random_state=0)
		kms.fit(X)
		
		return kms

	def hierachical(self, X, k):
        """
	top hierachical
        """
        linked = linkage(X, 'single')
        labelList = range(1, k)
        plt.figure(figsize=(10, 7))
        dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
        
        plt.show()
    
	def tsne(self, X, vocab):
	"""
	tsne을 이용한 2차원 축소
        """
        ts = TSNE(n_components=self.N_COMPONENTS)
		x_ts=ts.fit_transform(X)

		df = pd.DataFrame()

		font_name = font_manager.FontProperties(fname=self.FONT_DIR).get_name()
		rc('font', family=font_name)

		fig = plt.figure()
		fit.set_size_inches(40,20)
		ax = fig.add_subplot(1, 1, 1)
		ax.scatter(df['x'], df['y'])

		for word, pos in df.iterrows():
			ax.annotate(word, pos, fontsize=30)
		plt.show()

		return df
