import csv
from get_nps_from_articles import get_printable_str

'''
appends article headline to a csv which contains article id
'''

############

ART_to_TOPICS_FN = 'data/kaggle_election_day/election_2016_art_to_topics.csv'
ARTICLES_FN = 'data/kaggle_scrape_month/kaggle_scrape_month_12.csv'
HEADLINES_to_TOPICS_FN = 'data/kaggle_election_day/election_12_headlines_to_topics.csv'

############

aid_to_headline = {}
with open(ARTICLES_FN, 'rU') as f:
    freader = csv.DictReader(f)

    for line in freader:
        aid_to_headline[line['id']] = get_printable_str(line['title'])

with open(ART_to_TOPICS_FN, 'rU') as f:
    fieldnames = next(csv.reader(f))

fieldnames.append('headline')
g = open(HEADLINES_to_TOPICS_FN, 'w')
gwriter = csv.DictWriter(g, fieldnames = fieldnames)
gwriter.writeheader()

with open(ART_to_TOPICS_FN, 'rU') as f:
    freader = csv.DictReader(f)

    for line in freader:
        if line['article_id'] in aid_to_headline:
            line['headline'] = aid_to_headline[line['article_id']]
            gwriter.writerow(line)

g.close()
