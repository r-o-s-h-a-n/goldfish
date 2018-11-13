import csv
import sys
import spacy
import gensim
import ujson as json
from tqdm import tqdm


'''
Extracts noun phrases and cleans news articles.
'''

############
PROJECT_NAME = 'data/kaggle_scrape_sample'

HEADLINES_FN =  PROJECT_NAME+'.csv'
OUT_FN =        PROJECT_NAME+'_nps.json'

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

nlp = spacy.load('en')

STOP_WORDS = ('a', 'the', '\'s', '\t', 'who', 'what',
            'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
            'mon', 'tues', 'weds', 'thurs', 'fri',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sept', 'sep',
            'oct', 'nov', 'dec',
            'reporter'
            )

for word in STOP_WORDS:
    nlp.vocab[word].is_stop = True


############

def check_for_bad_np(np, remove_words):
    for w in remove_words:
        if w in np:
            return True
    return False

def process_text(text, remove_words):
    '''
    takes in a string of text and returns a list of lemmatized words
    '''
    text.lower()
    doc = nlp(text)

    nps = []
    for np in doc.noun_chunks:
        np = ' '.join([t.lemma_ for t in np if not t.is_stop])
        np.strip()

        # get rid of pesky pronouns
        if np == "-PRON-":
            continue

        if check_for_bad_np(np, remove_words):
            continue

        if np:
            nps.append(np)

    return nps

############
if __name__=='__main__':

    print('processing articles...')
    articles = {}

    with open(HEADLINES_FN, 'rU') as f:
        freader = csv.DictReader(f)

        for a in tqdm(freader):

            date = a['date']
            publication = a['publication'].lower()
            title = a['title'].lower()
            author = a['author'].lower()
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

            nps = process_text(N_HEADLINE_REPEATS*title+content, remove_words = (author, publication))
            articles[date][publication][a['id']] = nps

    with open(OUT_FN, 'w') as f:
        json.dump(articles, f)
