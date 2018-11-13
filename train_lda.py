import csv
import sys
import spacy
import gensim
from tqdm import tqdm
import ujson as json


'''
Generates keywords from articles with keyword weights. Uses an LDA Topic Model.
'''

############
PROJECT_NAME = 'data/kaggle_scrape_month/kaggle_scrape_month_1'

NPS_FN =        PROJECT_NAME+'_nps.json'
LDA_MODEL_FN =  PROJECT_NAME+'_LDA.mdl'
DCT_FN =        PROJECT_NAME+'_nps.dct'

NUM_TOPICS = 10     # num of topics for lda to generate
REMOVE_N = 20       # num of most frequent words to filter
NO_BELOW = 2        # num articles a phrase must appear in to be considered
NO_ABOVE = 0.6      # fraction of articles a phrase must appear in fewer than to be considered
############



############
print('loading saved extracted noun phrases...')
with open(NPS_FN, 'rb') as f:
    nps = json.load(f)

print('generating bow embeddings...')
corpus = [nps[date][publication][id] for date in nps for publication in nps[date] for id in nps[date][publication]]
print('num articles:')
print(len(corpus))

dct = gensim.corpora.Dictionary(corpus)
dct.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
dct.filter_n_most_frequent(REMOVE_N)
bow_corpus = [dct.doc2bow(article) for article in corpus]

print('training model...')
lda_model =  gensim.models.LdaModel(bow_corpus,
                                   num_topics = NUM_TOPICS,
                                   id2word = dct,
                                   minimum_probability = 0.1
                                   )

print('training complete. See topics generated:')

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

lda_model.save(LDA_MODEL_FN)
dct.save(DCT_FN)
