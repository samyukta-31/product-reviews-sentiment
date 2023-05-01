# Import packages
import gzip
import pandas as pd

# Function to parse the .json.gz file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

# Dunction to convert the .json.gz file into pandas dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Calling the function into a dataframe object
df = getDF('data\\reviews_Cell_Phones_and_Accessories_5.json.gz')

# Saving the dataframe to parquet file
df.to_parquet('data\\reviews_raw.parquet',index=False)