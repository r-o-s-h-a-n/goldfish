import os
import csv
import sys
import numpy as np
import ujson as json
from tqdm import tqdm

'''
use jaccard index on the keywords by publication to get similarity between topics reported by publications
'''

####################
PROJECT_NAME =  'data/kaggle_scrape_sample'
KEYWORDS_FN =   PROJECT_NAME+'_keywords.csv'
SIMILARITY_FN = PROJECT_NAME+'_similarity.csv'
######################

def get_similarity_metric(publication_to_keywords):
    '''
    calculates the number of keywords the publications share and divides by the
    total keywords the publications use. This ratio should be high when the
    publications talk about the same topics using the same keywords and low
    otherwise.
    '''
    k = set([w for p in publication_to_keywords for w in publication_to_keywords[p]])

    ocurrences = np.zeros(len(k))
    for i, w in enumerate(k):
        for p in publication_to_keywords:
            if w in publication_to_keywords[p]:
                ocurrences[i] += 1
    return sum(ocurrences)/len(ocurrences)

######################
g = open(SIMILARITY_FN, 'w')
gwriter = csv.DictWriter(g, fieldnames=['date', 'similarity_metric'])
gwriter.writeheader()

with open(KEYWORDS_FN, 'rU') as f:
    freader = csv.DictReader(f)

    curr_date = None
    for r in tqdm(freader):
        # assumes keywords are sorted by date
        if curr_date is None:
            publication_to_keywords = {}
            curr_date = r['date']

        if curr_date != r['date']:
            similarity = get_similarity_metric(publication_to_keywords)
            gwriter.writerow({'date':curr_date, 'similarity_metric':similarity})
            # calculate aggregates
            curr_date = r['date']
            publication_to_keywords = {}

        publication_to_keywords.setdefault(r['publication'], set()).add(r['keyword'])
