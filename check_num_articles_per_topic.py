import csv
from pprint import pprint


ARTICLES_TO_TOPICS_FN = 'data/kaggle_election_day/election_2016_art_to_topics.csv'

topic_count = {}

with open(ARTICLES_TO_TOPICS_FN, 'rU') as f:
    freader = csv.DictReader(f)

    for line in freader:
        topic = line['topic']

        if not topic in topic_count:
            topic_count[topic] = 0
        topic_count[topic] += 1

pprint(topic_count)
