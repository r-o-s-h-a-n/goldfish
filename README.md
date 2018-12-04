# goldfish

#### [Check out the visualization here](https://public.tableau.com/profile/vivek.kumar6595#!/vizhome/InformationBubbles/AllTopics)

The internet has changed the way the news works. We wanted to investigate if information bubbles exist and what they look like. In other words, if you were only reading news from one media outlet, we want to know when you were introduced to certain topics as compared to others using other media outlets.

Goldfish compares media attention to topics over time. Using the kaggle ["All the News" dataset](https://www.kaggle.com/snapcrack/all-the-news/home), we looked at 7 media outlets -- New York Times, Breitbart, Washington Post, CNN, Reuters, Buzzfeed News, Fox News -- from September 2016 through December 2016. We used the gensim [LDA topic model](https://radimrehurek.com/gensim/models/ldamodel.html) to generate topics that appeared in news articles. 



##  How it works

LDA infers a topic distribution for each article. For each publication and day, we summed each topic's inferred probability ("weight") over all articles for that publication on that day, and then normalized by number of articles on that day. This provided a ratio ranging from 0 to 1 measuring how much attention a publication was giving to each topic on each day.

In addition, for each publication, day, and topic, we found the article (by that publication and on that day) that had the highest probability of belonging to that topic. If the probability met a certain threshold, we visualized the article title in the tooltip of the visualization to provide context to the movements in topic prominence.




## How to run:

1. `pip install -r requirements.txt`
		
	* installs packages including gensim and spacy to your environment

2. Put the articles data in your `data/` directory.
	* get data from kaggle ["All the News" dataset](https://www.kaggle.com/snapcrack/all-the-news/home)


2. `get_nps_from_articles.py`

	* cleans articles and extracts noun phrases using spacy
	* removes most common words and words that don't appear in many different documents

3. `train_lda.py`

	* trains LDA topic model using noun phrases 

4. `get_article_topic.py`

	*  assigns topics to articles and topics to publications on a given day
	*  This creates the data file that is used to plot the topic prominence over time for each publication
	
5. `link_aid_to_headline.py`
	* creates csv containing article titles ("headlines"), their topics, and topic weights for each publication
	* This is the data file that is used to visualize the headlines in tooltip

6. Plot data
	* Download the [tableau view](https://public.tableau.com/profile/vivek.kumar6595#!/vizhome/InformationBubbles/AllTopics) we created. 
