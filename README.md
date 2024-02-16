README
==============================

### Cluster-wise Topic-based Sentiment Analysis for Amazon Cell Phone and Accessories Product Reviews

Project Organization
------------

    ├── LICENSE
    |
    ├── README.md                   <- The top-level README for viewers and users of this project.
    |
    ├── data                        <- Zipped, Raw, preprocessed and clustered datasets are stored here
    |
    ├── models                      <- Trained and serialized cluster-wise LDA model along with metadata
    |
    ├── notebooks                   <- Jupyter notebooks for exploration
    |   |                   
    │   └── clustering_visualization.ipynb
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── src                         <- Source code for use in this project.
    |   |
    │   ├── read_data.py            <- Unzips and reads data from local, processes it into dataframe and stores as parquet (taken from the source API website https://jmcauley.ucsd.edu/data/amazon/ )
    │   │
    │   ├── preprocessing.py        <- Text preprocessing for cluster, topic modelling and sentiment analysis  
    │   │
    │   ├── clustering.py           <- K-Means clustering for all product reviews (into 4 clusters)
    │   │
    │   ├── topic_modelling.py      <- Cluster-wise topic modelling to find predominent themes in similar reviews
    │   │
    │   └── sentiment_analysis.py   <- Sentiment analysis of the clustered reviews
    │
    └── run.sh                      <- Bash file to all files in sequence

---

### Run

Open bash command line and run the following

```
.\run.sh
``` 

### Data Source

https://jmcauley.ucsd.edu/data/amazon/ 

### Step-wise project details

* General preprocessing of reviews following
   * Lowercase conversion
   * Lemmatisation
   * Stop work removal
   * Punctuation removal

* Vectorizing using TfIdf for Clustering

* Clustering using KMeans into 4 clusters

* Pre-processing prior to topic modelling
   * Removal of new-line characters
   * Removal of quotation marks
   * Removal of filler words
   * Removal of numbers
   * Removal of extra spaces
   * Removal of words with less than 3 letters

* Training LDA model for each cluster to get the predominant topics for that topic

image.png

* Performing sentiment analysis for each cluster to understand the sentiment associated with said topics


