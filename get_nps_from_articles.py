import csv
import sys
import spacy
import gensim
import ujson as json
from tqdm import tqdm
import random
import string
printable = set(string.printable)

'''
Extracts noun phrases and cleans news articles.
'''

############
PROJECT_NAME = 'data/kaggle_election_day/election_2016'

HEADLINES_FNS =  ['data/kaggle_scrape_month/kaggle_scrape_month_9.csv'
                , 'data/kaggle_scrape_month/kaggle_scrape_month_10.csv'
                , 'data/kaggle_scrape_month/kaggle_scrape_month_11.csv'
                , 'data/kaggle_scrape_month/kaggle_scrape_month_12.csv']
OUT_FN =        PROJECT_NAME+'_nps.json'

# To give headlines more weight in training and prediction, this var selects how
# many times you want to repeat the headline in the text
N_HEADLINE_REPEATS = 10

PUBS = ('New York Times'
        , 'Breitbart'
        , 'Reuters'
        , 'CNN'
        , 'Washington Post'
        , 'Fox News'
        , 'Buzzfeed News'
        )

############

csv.field_size_limit(sys.maxsize)

# nlp = spacy.load('en')
nlp = spacy.load('en_core_web_md')

STOP_WORDS = ('a', 'the', '\'s', '\t', 'who', 'what',
            'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
            'mon', 'tues', 'weds', 'thurs', 'fri',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sept', 'sep',
            'oct', 'nov', 'dec',
            'reporter', 'ms', 'mr'
            )

for word in STOP_WORDS:
    nlp.vocab[word].is_stop = True


############
def get_phrase_lemma(phrase):
    if isinstance(phrase, str):
        phrase = nlp(phrase)
    return ' '.join([t.lemma_ for t in phrase if (not t.is_stop and t.pos_ != 'PRON')])

def check_for_bad_np(np, remove_words):
    for w in remove_words:
        if w in np:
            return True
    return False

def get_printable_str(s):
    c = filter(lambda x: x in printable, s)
    return ''.join([i for i in c])

def process_text(text, remove_words=[]):
    '''
    takes in a string of text and returns a list of lemmatized words
    '''
    text.lower()
    text = get_printable_str(text)
    doc = nlp(text)

    nps = []
    for np in doc.noun_chunks:
        np = get_phrase_lemma(np)
        np.strip()

        if check_for_bad_np(np, remove_words):
            continue

        if np:
            nps.append(np)

    return nps

############
if __name__=='__main__':

    print('processing articles...')
    articles = {}

    for fn in HEADLINES_FNS:
        with open(fn, 'rU') as f:
            freader = csv.DictReader(f)

            for a in tqdm(freader):

                date = a['date']
                publication = a['publication'].lower()
                title = a['title'].lower()
                author = a['author'].lower().strip()
                content = a['content'].lower()

                if not date in articles:
                    articles[date] = {
                            'new york times': {}
                            , 'breitbart': {}
                            , 'reuters': {}
                            , 'cnn': {}
                            , 'washington post': {}
                            , 'fox news': {}
                            , 'buzzfeed news': {}
                            }

                text = '. '.join([title]*N_HEADLINE_REPEATS) + content
                remove_words = [c for c in [get_phrase_lemma(author)
                                            , get_phrase_lemma(publication)]
                                    if c]

                nps = process_text(text, remove_words = remove_words)

                articles[date][publication][a['id']] = nps

    with open(OUT_FN, 'w') as f:
        json.dump(articles, f)

    corpus = [articles[date][publication][id] for date in articles for publication in articles[date] for id in articles[date][publication]]
    print('num articles:')
    print(len(corpus))
