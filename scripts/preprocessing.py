# Import Packages
import nltk
import string
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download nltk packages for specific preprocessing tasks
nltk.download('punkt')
nltk.download('wordnet') 
nltk.download('omw-1.4')
nltk.download("stopwords")

# Read parquet file into pandas dataframe
reviews = pd.read_parquet('data\\reviews_raw.parquet')

# Define lemmatization and stopwords objects
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

# Convert reviewsText column to string
reviews["reviewText"] = reviews["reviewText"].astype('str')

# Create duplicate reviewText column called reviewTokens to perform preprocessing
reviews["reviewTokens"] = reviews["reviewText"]

# Convert all words in each review to lowercase
reviews["reviewTokens"] = reviews["reviewTokens"].apply(lambda a: a.lower())

# Remove all punctuations from each review
reviews["reviewTokens"] = reviews["reviewTokens"].str.replace('[{}]'.format(string.punctuation), '')

# Remove all stopwords from each review
reviews["reviewTokens"] = reviews["reviewTokens"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# Perform lemmatization on each word of each review
reviews["reviewTokens"] = reviews["reviewTokens"].apply(lambda a: lemmatizer.lemmatize(a))

# Split the words in each review into a list of words by tokenization
# reviews["reviewTokens"] = reviews["reviewTokens"].apply(word_tokenize)

# # Combine words occuring together as bigrams in a last for each review
# reviews['reviewTokens_bigrams'] = reviews["reviewTokens"].apply(lambda row: list(nltk.ngrams(row, 2))) 

# # Combine words occuring together as unigrams in a last for each review
# reviews['reviewTokens_unigrams'] = reviews["reviewTokens"].apply(lambda row: list(nltk.ngrams(row, 1))) 

# Save preprocessed file with reviewTokens column to parquet file
reviews.to_parquet("data\\reviews_preprocessed.parquet")