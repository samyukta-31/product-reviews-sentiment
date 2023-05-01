# perform topic modelling to find the most prominent topic among all the words in each cluster

import pandas as pd
import re
import string
import nltk
import gensim
import gensim.corpora as corpora

nltk.download('averaged_perceptron_tagger')

# read the data
df = pd.read_parquet('data\\reviews_clustered.parquet')

# setting counter for each model
counter = 0

# making sub-dataframes for each cluster
cluster0 = df[df['cluster'] == 0]
cluster1 = df[df['cluster'] == 1]
cluster2 = df[df['cluster'] == 2]
cluster3 = df[df['cluster'] == 3]

for cluster in [cluster0, cluster1, cluster2, cluster3]:
    data = cluster.reviewTokens.values.tolist()

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # Remove filler words
    data = [re.sub("like", "", sent) for sent in data]
    data = [re.sub("would", "", sent) for sent in data]
    data = [re.sub("get", "", sent) for sent in data]
    data = [re.sub("one", "", sent) for sent in data]
    data = [re.sub("really", "", sent) for sent in data]
    data = [re.sub("even", "", sent) for sent in data]
    data = [re.sub("also", "", sent) for sent in data]
    data = [re.sub("much", "", sent) for sent in data]
    data = [re.sub("good", "", sent) for sent in data]
    data = [re.sub("great", "", sent) for sent in data]
    data = [re.sub("well", "", sent) for sent in data]
    data = [re.sub("dont", "", sent) for sent in data]
    data = [re.sub("make", "", sent) for sent in data]
    data = [re.sub("made", "", sent) for sent in data]

    # Remove punctuation
    data = [re.sub('[%s]' % re.escape(string.punctuation), '', sent) for sent in data]

    # Remove numbers
    data = [re.sub('\w*\d\w*', '', sent) for sent in data]

    # Remove extra spaces
    data = [re.sub(' +', ' ', sent) for sent in data]

    # Split each sentence into words
    data = [sent.split() for sent in data]

    # Remove words with length less than 3
    data = [[word for word in sent if len(word) > 2] for sent in data]

    # remove adjective, verbs and adverbs
    # data = [[word for word in sent if 
    #              nltk.pos_tag([word])[0][1] not in ('JJ', 'JJR', 'JJS', 'VBP', 'VBD', 'VBN', 'VBG', 'RB', 'RBR', 'RBS')] 
    #              for sent in data]

    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus
    texts = data

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 2

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=0)

    # save the model
    lda_model.save(f'models\\lda_model_{counter}.model')

    # incrementing counter
    counter += 1
