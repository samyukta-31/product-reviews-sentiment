import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_processed = pd.read_parquet('data\\reviews_preprocessed.parquet')

vect = CountVectorizer()

df_processed['unigrams_flattened'] = [ng for unigram in df_processed['reviewTokens_unigrams'] for ng in unigram]

vect.fit(df_processed['unigrams_flattened'])

cluster = KMeans(n_clusters = 3)
cluster_labels = cluster.fit_predict(vect)

print(cluster_labels)

plt.scatter(cluster_labels[:,0])