import csv
import sys
import spacy
import gensim
import random
from tqdm import tqdm
import ujson as json


'''
Generates keywords from articles with keyword weights. Uses an LDA Topic Model.
'''

############
PROJECT_NAME = 'data/kaggle_election_day/election_2016'

NPS_FN =        PROJECT_NAME+'_nps.json'
LDA_MODEL_FN =  PROJECT_NAME+'_LDA.mdl'
DCT_FN =        PROJECT_NAME+'_nps.dct'

NUM_TOPICS = 20         # num of topics for lda to generate
REMOVE_N = 40           # num of most frequent words to filter
NO_BELOW = 3            # num articles a phrase must appear in to be considered
NO_ABOVE = 0.6          # fraction of articles a phrase must appear in fewer than to be considered
NORMALIZE_PUBS = True   # train with an equal number of articles per publication by repeating some articles
############



############
print('loading saved extracted noun phrases...')
with open(NPS_FN, 'rb') as f:
    nps = json.load(f)

if NORMALIZE_PUBS:
    print('normalizing number of articles by publication...')
    pubs_to_nps = {}

    for date in nps:
        for publication in nps[date]:
            if publication not in pubs_to_nps:
                pubs_to_nps[publication] = []
            for id in nps[date][publication]:
                pubs_to_nps[publication].append(nps[date][publication][id])

    n_articles = {p: len(pubs_to_nps[p]) for p in pubs_to_nps}
    max_articles = max(list(n_articles.values()))

    for p in pubs_to_nps:
        repeat_articles = pubs_to_nps[p]*(max_articles//n_articles[p])
        repeat_articles.extend(random.sample(pubs_to_nps[p], max_articles%n_articles[p]))
        pubs_to_nps[p].extend(repeat_articles)

    corpus = [a for p in pubs_to_nps for a in pubs_to_nps[p]]

else:
    corpus = [nps[date][publication][id] for date in nps for publication in nps[date] for id in nps[date][publication]]

print('num articles:')
print(len(corpus))

print('generating bow embeddings...')
dct = gensim.corpora.Dictionary(corpus)
dct.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
dct.filter_n_most_frequent(REMOVE_N)
bow_corpus = [dct.doc2bow(article) for article in corpus]

print('training model...')
lda_model =  gensim.models.LdaModel(bow_corpus,
                                   num_topics = NUM_TOPICS,
                                   id2word = dct,
                                   minimum_probability = 0.1,
                                   passes = 15
                                   )

print('training complete. See topics generated:')

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

lda_model.save(LDA_MODEL_FN)
dct.save(DCT_FN)
