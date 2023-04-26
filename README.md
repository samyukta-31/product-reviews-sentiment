# product-reviews-sentiment
Finding the most impactful features determining sentiment of Amazon Cell Phones and Accessories reviews

## How to execute this Program

### Run
Open bash command line and run the following

```
.\run.sh
```

## Concept: Researching a Topic-based approach to Sentiment Analysis using Graphs for product reviews

### Data Source
https://jmcauley.ucsd.edu/data/amazon/ 

### Steps followed to pursue this research

1. Preprocessing of reviews and converting to bigrams
2. Clustering of all tokens
2. Topic modelling of individual clusters - collection of topics is kept as central nodes for graph based analysis
5. Graph-based sentiment analysis with topics as central nodes
6. Calculation of sentiment score based on associated nodes (binary classification)
7. Prediction would be overall sentiment corresponding to influential topics
