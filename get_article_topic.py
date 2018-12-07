import csv
import gensim
import ujson as json
from tqdm import tqdm


'''
predicts most prominent topic for a set of articles
'''

############
MODEL_NAME = 'data/kaggle_election_day/election_2016'
PROJECT_NAME = 'data/kaggle_election_day/election_2016'

LDA_MODEL_FN =  MODEL_NAME+'_LDA.mdl'
DCT_FN =        MODEL_NAME+'_nps.dct'
NUM_TOPICS = 20

NPS_FN =        PROJECT_NAME+'_nps.json'
ART_to_TOPICS_FN = PROJECT_NAME+'_art_to_topics.csv'
PUB_to_TOPICS_FN = PROJECT_NAME+'_pub_to_topics.csv'

############


print('Loading saved LDA model...')
lda_model = gensim.models.LdaModel.load(LDA_MODEL_FN)

print('Loading saved Dictionary...')
dct = gensim.corpora.Dictionary.load(DCT_FN)

print('Generating and saving topics for each article...')
with open(NPS_FN, 'rb') as f:
    nps = json.load(f)

g = open(ART_to_TOPICS_FN, 'w')
gwriter = csv.DictWriter(g, fieldnames = ['date', 'publication', 'article_id', 'topic', 'article_to_topic_weight'])
gwriter.writeheader()

h = open(PUB_to_TOPICS_FN, 'w')
hwriter = csv.DictWriter(h, fieldnames = ['date', 'publication', 'topic', 'topic_to_pub_weight'])
hwriter.writeheader()

sorted_dates = sorted(nps)
sorted_publications = sorted(nps[sorted_dates[0]])
all_topics = range(0, NUM_TOPICS)

for date in tqdm(sorted_dates):
    for publication in sorted_publications:

        topic_to_pub_weight = {}
        # normalize the weight of each topic by the number of articles
        n_articles = len(nps[date][publication])

        last_id = -1
        for id in nps[date][publication]:
            if id == last_id:
                print(id)
            last_id = id

            article = nps[date][publication][id]
            prediction = lda_model[dct.doc2bow(article)]

            for t in prediction:
                topic = t[0]
                all_topics.add(topic)
                art_to_topic_weight = t[1]

                if not topic in topic_to_pub_weight:
                    topic_to_pub_weight[topic] = 0
                topic_to_pub_weight[topic] += art_to_topic_weight

                out = {'date': date
                    , 'publication': publication
                    , 'article_id': id
                    , 'topic': topic
                    , 'article_to_topic_weight': art_to_topic_weight
                    }

                gwriter.writerow(out)

        sum_topic_weights = sum(list(topic_to_pub_weight.values()))

        unseen_topics = [t for t in all_topics if t not in topic_to_pub_weight]
        for topic in sorted(topic_to_pub_weight, key=topic_to_pub_weight.__getitem__, reverse=True):

            out = {'date': date
                , 'publication': publication
                , 'topic': topic
                , 'topic_to_pub_weight': float(topic_to_pub_weight[topic])/sum_topic_weights
                }

            hwriter.writerow(out)

g.close()
h.close()
