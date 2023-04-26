import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class Cluster:
    def kmeans(self):
        # cluster all words in reviewsTokens column

        df_processed = pd.read_parquet('data\\reviews_preprocessed.parquet')
        # Create tfidf vectorizer object
        vectorizer = TfidfVectorizer()

        # Create tfidf matrix
        tfidf_matrix = vectorizer.fit_transform(df_processed["reviewTokens"])

        # Create kmeans object
        kmeans = KMeans(n_clusters=4, random_state=0)

        # Fit kmeans object to tfidf matrix
        kmeans.fit(tfidf_matrix)

        # Create cluster column in dataframe
        df_processed["cluster"] = kmeans.labels_

        df_processed.to_parquet("data\\reviews_clustered.parquet")

if __name__ == "__main__":
    cluster = Cluster()
    cluster.kmeans()



