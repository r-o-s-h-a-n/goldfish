import csv
import sys
import spacy
import gensim
from tqdm import tqdm
import ujson as json
import numpy as np
from scipy.sparse import csr_matrix, vstack


'''
Generates keywords from articles using already trained LDA model
'''

############
MODEL_NAME = 'data/kaggle_scrape_month/kaggle_scrape_month_1'
# MODEL_NAME =    'data/kaggle_scrape_sample'
PROJECT_NAME =  'data/kaggle_scrape_sample'

LDA_MODEL_FN =  MODEL_NAME+'_LDA.mdl'
DCT_FN =        MODEL_NAME+'_nps.dct'
NPS_FN =        PROJECT_NAME+'_nps.json'
KEYWORDS_FN =   PROJECT_NAME+'_keywords.csv'

N_TOP_KEYWORDS = 50



############

def convert_sparse_matrix_to_csr(s, n_elements):
    '''
    converts a sparse 1-d matrix as a list of tuples to a scipy csr sparse matrix
    '''
    row_ind = np.array([0 for t in s])
    col_ind = np.array([t[0] for t in s])
    data = np.array([t[1] for t in s])
    return csr_matrix((data, (row_ind, col_ind)), shape=(1, n_elements))


def get_normalized_topic_vector(bow_vector, lda_model):
    '''
    generate vector embedding of latent topics discussed by a given
    publication on given date
    '''
    topics = lda_model[bow_vector]
    num_topics = lda_model.num_topics
    topics = convert_sparse_matrix_to_csr(topics, num_topics)
    topics = topics.multiply(1/topics.max()) # normalize linearly
    return topics

def get_square_bow_filter_matrix(bow_vector, dct):
    '''
    generate a square matrix of bow to filter words in the vocabulary
    by the words that actually ocurred in the publication on the date
    '''
    size_vocabulary = len(dct.keys())
    bow = convert_sparse_matrix_to_csr(bow_vector, size_vocabulary)
    bow = bow.multiply((1/bow.max())) # normalize linearly
    bow = vstack([bow for _ in range(size_vocabulary)])
    return bow

############

print('Loading saved LDA model...')
lda_model = gensim.models.LdaModel.load(LDA_MODEL_FN)
# term_topic is the probability for each word in each topic, shape (num_topics, vocabulary_size)
term_topic = csr_matrix(lda_model.get_topics())
term_topic = term_topic.multiply(1/term_topic.max()) # normalize linearly


print('Loading saved Dictionary...')
dct = gensim.corpora.Dictionary.load(DCT_FN)


print('Generating and saving keywords for each article...')
with open(NPS_FN, 'rb') as f:
    nps = json.load(f)

g = open(KEYWORDS_FN, 'w')
gwriter = csv.DictWriter(g, fieldnames=['date', 'publication', 'keyword', 'weight'])
gwriter.writeheader()

sorted_dates = sorted(nps)
sorted_publications = sorted(nps[sorted_dates[0]])

keywords = {}
for date in tqdm(sorted_dates):
    for publication in tqdm(sorted_publications):

        # vector which will store the sum of keyword weights for a given
        # publication on a given date
        k = None

        for id in nps[date][publication]:

            phrases = nps[date][publication][id]
            bow = dct.doc2bow(phrases)

            if not any(bow):
                continue

            topics = get_normalized_topic_vector(bow, lda_model)
            bow = get_square_bow_filter_matrix(bow, dct)

            # multiply the topics, term_topic, and bow to get a metric of
            # keyword importance in the document
            if k is None:
                k = topics @ term_topic @ bow
            else:
                k += topics @ term_topic @ bow

        if k is None:
            continue

        # find the top keywords that correspond to the highest weights and save
        weights = k.toarray().transpose().tolist()

        for w in sorted(weights, reverse=True)[:N_TOP_KEYWORDS]:
            idx = weights.index(w)

            gwriter.writerow({'date': date
                            , 'publication': publication
                            , 'keyword': dct.id2token[idx]
                            , 'weight': w[0]
                            })

g.close()
