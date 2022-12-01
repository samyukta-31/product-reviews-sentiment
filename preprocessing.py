import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
reviews = pd.read_csv('data\\reviews.csv')

reviews.info()

reviews["reviewText"] = reviews["reviewText"].astype('str')
reviews["reviewTokens"] = reviews["reviewText"].apply(word_tokenize)

# reviews.head(10)