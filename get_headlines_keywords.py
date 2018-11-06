import os
import csv
import sys
import json
import math
import spacy
import string
import datetime
import itertools
import collections
from sklearn.feature_extraction.text import TfidfVectorizer


csv.field_size_limit(sys.maxsize)
nlp = spacy.load('en')

HEADLINES_FN = 'data/kaggle_scrape_big_sample.csv'

STOP_WORDS = ('a', 'the', '\'s', '-PRON-', '\t', 'who', 'what')

ZIPFS_N = 0

CONTENT_CHAR_LIMIT = 1000

N_HEADLINE_REPEATS = 2

# extract headlines and top 1000 characters of article

# train keyword model
    # extract noun phrases using spacy dep tree
    # tfidf?
    # get nerts?
# predict keywords and metric
# save date, news site, keywords to metrics and article id json dumped dicts



#######################################################
def get_idfs(corpus):
    idf_values = {}
    all_tokens_set = set([item for sublist in corpus for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, corpus)
        idf_values[tkn] = 1 + math.log(len(corpus)/(sum(contains_token)))
    return idf_values

#######################################################

print('extracting noun phrases from articles...')
with open(HEADLINES_FN, 'rU') as f:
    f_reader = csv.DictReader(f)

    examples = {}
    dates_to_articles = {}

    for i, a in enumerate(f_reader):
        # if i == 100:
        #     break
        publication = a['publication'].lower()
        text = ' . '.join([N_HEADLINE_REPEATS*(a['title']+'. '), a['content'][:CONTENT_CHAR_LIMIT]])
        doc = nlp(text)

        nps = []
        for np in doc.noun_chunks:
            np = ' '.join([t.lemma_ for t in np if t.lemma_ not in STOP_WORDS])
            np.strip()
            if publication in np or a['author'].lower() in np:
                continue
            if np:
                nps.append(np)

        examples[a['id']] = nps
        if not a['date'] in dates_to_articles:
            dates_to_articles[a['date']] = {}
        if not publication in dates_to_articles[a['date']]:
            dates_to_articles[a['date']][publication] = {'ids':[], 'tfidfs': None}
        dates_to_articles[a['date']][publication]['ids'].append(a['id'])


zipfs_words = []
if ZIPFS_N != 0:
    print('using zipfs law to find most common noun phrases for removal. Num of words to remove: ', str(ZIPFS_N))
    corpus = list(examples.values())
    corpus = list(itertools.chain.from_iterable(corpus))
    word_freqs = collections.Counter(corpus)
    zipfs_words = sorted(word_freqs, key=word_freqs.get, reverse=True)[:ZIPFS_N]
    print('words exclude because of zipfs law:')
    print(zipfs_words)


print('calculating idf values for phrases in corpus')
corpus = []
ids = sorted(examples.keys())
for id in ids:
    for p in examples[id]:
        if p in zipfs_words:
            examples[id].remove(p)
    corpus.append(examples[id])

idfs = get_idfs(corpus)

print('idf of `trump`:', idfs['trump'])
print('idf of `disney`:', idfs['disney'])


# write top tfidfs to csv
print('finding tfidf values and writing to csv')
with open('data/top_words.csv', 'w') as f:
    f_writer = csv.writer(f)
    f_writer.writerow(['date', 'publication', 'ids', 'top_words'])

    for date in dates_to_articles:
        for publication in dates_to_articles[date]:
            phrases = []
            for id in dates_to_articles[date][publication]['ids']:
                phrases.extend(examples[id])
            word_freqs = collections.Counter(phrases)
            tfidfs = {w: idfs[w]*word_freqs[w] for w in word_freqs}
            top_words = sorted(tfidfs, key=tfidfs.get, reverse=True)[:20]
            # top_tfidfs = {w: tfidfs[w] for w in top_words}
            dates_to_articles[date][publication]['tfidfs'] = top_words

            f_writer.writerow([date
                            , publication
                            , json.dumps(ids)
                            , json.dumps(top_words)
                            ])
