# product-reviews-sentiment
Finding the most impactful features determining sentiment of Amazon Cell Phones and Accessories reviews

## How to execute this Program

### Dependencies

```
pandas==1.4.2
nltk==3.7
```
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
2. Topic modelling of individual clusters - collection of topics is kept as corpus for user input categories
3. Preferred category is taken as dynamic user input via API
4. Network Graph is created based on bigram occurances as weights and bigrams as nodes
5. Graph-based sentiment analysis with chosen category as central node
6. Calculation of sentiment score based on associated nodes (binary classification)
7. Prediction would be overall sentiment of that category