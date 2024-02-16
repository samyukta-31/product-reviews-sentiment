import gensim 
import pandas as pd
from transformers import pipeline

df = pd.read_parquet('data\\reviews_clustered.parquet')

# read lda model
for i in range(0,4):
    lda_model = gensim.models.ldamodel.LdaModel.load(f'models\\lda_model_{i}.model')

    # making sub-dataframes for each cluster
    cluster0 = df[df['cluster'] == i]

    # print the topics with largest weights
    topics = list(lda_model.print_topics())
    individual_topics = []
    for topic in topics:
        topic_words = topic[1].split(' + ')
        for topic_word in topic_words:
            topic_word = topic_word.split('*')[1]
            topic_word = topic_word.replace('"', '')
            if topic_word not in individual_topics:
                individual_topics.append(topic_word)
    print(f'Cluster {i}: {individual_topics}')

    # Find sentiment for reviews corresponding to each topic
    for topic in individual_topics:
        topic_reviews = cluster0[cluster0['reviewTokens'].str.contains(topic)]
        topic_reviews = topic_reviews[['reviewText', 'reviewTokens']]

        # sentiment analysis using huggingface BERT
        sentiment_analysis = pipeline('sentiment-analysis')
        
        topic_reviews['sentiment'] = topic_reviews['reviewText'].apply(lambda x: sentiment_analysis(x[:513])[0]['label'])

        # Visualize sentiment
        print(topic)
        print(topic_reviews['sentiment'].value_counts())
        print('\n')


    