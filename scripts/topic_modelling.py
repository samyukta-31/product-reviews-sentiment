# perform topic modelling to find the most prominent topic among all the words in each cluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# read the data
df = pd.read_parquet('data\\reviews_clustered.parquet')

# Create tfidf vectorizer object
vectorizer = TfidfVectorizer()

# perform topic modelling for reviewTokens in each cluster

# making sub-dataframes for each cluster
cluster0 = df[df['cluster'] == 0]
cluster1 = df[df['cluster'] == 1]
cluster2 = df[df['cluster'] == 2]
cluster3 = df[df['cluster'] == 3]

# Create tfidf matrix
clust0_vect = vectorizer.fit_transform(cluster0["reviewTokens"])
clust1_vect = vectorizer.fit_transform(cluster1["reviewTokens"])
clust2_vect = vectorizer.fit_transform(cluster2["reviewTokens"])
clust3_vect = vectorizer.fit_transform(cluster3["reviewTokens"])

# Create LDA object
lda = LatentDirichletAllocation(n_components=2, random_state=0)

# Fit LDA object to tfidf matrix
clust0_lda = lda.fit_transform(clust0_vect)
clust1_lda = lda.fit_transform(clust1_vect)
clust2_lda = lda.fit_transform(clust2_vect)
clust3_lda = lda.fit_transform(clust3_vect)

# Create cluster column in dataframe
# cluster0["topic"] = clust0_lda.argmax(axis=1)
# cluster1["topic"] = clust1_lda.argmax(axis=1)
# cluster2["topic"] = clust2_lda.argmax(axis=1)
# cluster3["topic"] = clust3_lda.argmax(axis=1)

# print(cluster0['topic'].head(5))


def display_topics(model, feature_names):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-2 - 1:-1]]))


# display_topics(clust0_lda, clust0_vect.get_feature_names())
# 



