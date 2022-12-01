import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
reviews = pd.read_csv('data\\reviews.csv', converters={'helpful': pd.eval})

reviews.info()
print(reviews[reviews['reviewText']==''])
reviews["reviewTokens"] = reviews["reviewText"].apply(lambda x: nlp.tokenizer(x))

reviews.head(10)