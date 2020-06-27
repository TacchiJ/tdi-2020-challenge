#!/usr/bin/env python
# coding: utf-8

# # Project Proposal

# ## Idea: Sentiment analysis of Corona Virus Tweets compared against it's spread, by state

# Propose a project that uses a large, publicly accessible dataset. Explain your motivation for tackling this problem, discuss the data source(s) you are using, and explain the analysis you are performing. At a minimum, you will need to do enough exploratory data analysis to convince someone that the project is viable and generate two interesting non-trivial plots or other assets supporting this. Explain the plots and give url links to them.

# #### Justification:
# Since the outbreak of the COVID-19 pandemic, numerous problems have arisen due to the spreading of fake news and misinformation. These problems are changing peoples behaviours and costing lives. I propose a look into how shared information about COVID-19 is shared and how this affects peoples' attitudes and behaviours, and consequently the spread of the virus.
# 
# I have conducted an investigatory analysis using time series data on the confirmed cases of coronavirus in the US, provided by the John Hopkins University. I have paired this with a Naive Bayes sentiment analysis of data from Official US Government Twitter account (state-level). 
# 
# Twitter is a rich and up-to-date source of information that comes directly from the public. Twitter is also used by many official bodies to help share important information. It is therefore ideal for getting an impression of how information is shared and the effects this has on peoples' attitudes towards COVID-19.

# #### Data sources:
# * Confirmed Data: https://github.com/CSSEGISandData/COVID-19
# * Death Data: https://github.com/CSSEGISandData/COVID-19
# * Tweet Data: https://developer.twitter.com/en

# ### Installs and imports

# In[1]:


def run_installs():
    get_ipython().system('pip install nltk')
    get_ipython().system('pip install pandas')
    get_ipython().system('pip install seaborn')
    get_ipython().system('pip install tweepy')
    
# Uncomment next line to install required libraries
# run_installs()


# In[90]:


import asyncio
import collections
import datetime
import math
import os
import random
import re
import string
import sys
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tweepy

from nltk import classify
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[91]:


sns.set()


# ## Virus Data

# In[3]:


if not os.path.exists('data'):
    os.makedirs('data')


# In[4]:


confirmed_df = pd.read_csv('data/time_series_covid19_US.csv', low_memory=False)  # 1.6 MB


# In[5]:


confirmed_df.head(5)


# ### Limit data to just the 48 US mainland states 
# 
# This should minimize the affects of cultural differences or any other confounding factors.

# In[6]:


states = set(confirmed_df['Province_State'])
excluded_regions = ['Grand Princess',
                    'Alaska',
                    'Northern Mariana Islands',
                    'Puerto Rico',
                    'Guam',
                    'Hawaii',
                    'American Samoa',
                    'District of Columbia',
                    'Virgin Islands',
                    'Diamond Princess']

def exclude_regions(regions):
    return [True if r not in excluded_regions else False for r in regions]


# In[7]:


states_codes = {'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida',
                'GA': 'Georgia', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
                'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts',
                'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
                'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
                'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
                'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
                'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia',
                'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
len(states_codes)


# In[8]:


US_mainland_df = confirmed_df[exclude_regions(confirmed_df['Province_State'])]
mainland_states = sorted(set(US_mainland_df['Province_State']))
print(len(mainland_states))
US_mainland_df.head(10)


# In[9]:


start_date = '4/1/20'  # Start from 1st April 2020
sum_by_state = {state: US_mainland_df[US_mainland_df['Province_State'] == state].sum(axis = 0, skipna = True) for state in mainland_states}
US_df = pd.DataFrame(sum_by_state).loc[start_date:]
US_df


# ### Engineer Feature: Increase from prevous day 

# In[10]:


increase_df = pd.DataFrame({row1[0]: row1[1] - row0[1] for row1, row0 in zip(US_df[1:].iterrows(), US_df[:-2].iterrows())}).T
increase_df


# ## Twitter Data

# My Twitter credentials have been redacted from this notebook. However, a csv file containing all the information that was used has been included in its stead.

# ### Import data from csv

# In[20]:


# all_gov_tweets_df = pd.read_csv(f"data/gov_tweets_final.csv")


# #### Authorization

# In[11]:


credentials = dict(pd.read_csv('data/twitter_credentials.csv', low_memory=False))
key = credentials['TWITTER_KEY'][0]
secret_key = credentials['TWITTER_SECRET_KEY'][0]
access_token = credentials['TWITTER_ACCESS_TOKEN'][0]
access_token_secret = credentials['TWITTER_ACCESS_TOKEN_SECRET'][0]


# In[12]:


auth = tweepy.OAuthHandler(key, secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# #### Search API

# In[13]:


columns=['query', 'user', 'text', 'date', 'location', 'link']

def get_tweets(query, n=10, lang='en'):
    tweets = []
    for tweet in api.search(q=query, count=n, lang=lang):
        data = {columns[0]: query,
                columns[1]: tweet.user.name,
                columns[2]: tweet.text,
                columns[3]: tweet.created_at,
                columns[4]: tweet.user.location,
                columns[5]: f"https://twitter.com/{tweet.user.id}/status/{tweet.id}"}
        tweets.append(data)
    return tweets


# In[14]:


queries = ['corona', 'virus', 'washing hands', 'doctor', 'hospital', 'government']
max_tweets = 100
tweet_df = pd.DataFrame(get_tweets(queries[0], max_tweets))
for query in queries[1:]:
    tweets = get_tweets(query, max_tweets)
    tweet_df = tweet_df.append(tweets)
tweet_df


# Unfortunately, the Twitter search API is limited to the past 7 days. If we want to go beyond that, we will have to either pay for the Premium API or use a user_timeline. I will go for the latter option.

# #### User TImeline

# As this alternative method only allows for the tweets of specific users to be searched for, I have decided to use state officals as they regularly release updates and presumably have at least some influence over the people in their state. 
# 
# Only verified accounts were included and not all states had verified government accounts.

# In[15]:


states_codes = {'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida',
                'GA': 'Georgia', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
                'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts',
                'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
                'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
                'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
                'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
                'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia',
                'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
len(states_codes)


# In[191]:


target_users = {'CAgovernor': 'California',
                'delaware_gov': 'Delaware',
                'georgiagov': 'Georgia',
                'IN_gov': 'Indiana',
                'ksgovernment': 'Kansas',
                'StateMaryland': 'Maryland',
                'CarsonCityGov': 'Nevada',
                'NYGov': 'New York',
                'rigov': 'Rhode Island',
                'texasgov': 'Texas',
                'UtahGov': 'Utah',
                'WAStateGov': 'Washington'}
user_tweets = []


# In[17]:


def get_user_timeline_tweets(user, n=10):
    print(user)
    for status in tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode="extended").items(n):
        tweet = status._json
        data = {columns[1]: tweet['user']['screen_name'],
                columns[2]: tweet['full_text'],
                columns[3]: tweet['created_at'],
                columns[4]: tweet['user']['location'],
                columns[5]: f"https://twitter.com/{tweet['user']['id']}/status/{tweet['id']}"}
        user_tweets.append(data)


# In[18]:


num_tweets = 2500
for t_user in target_users:
    get_user_timeline_tweets(t_user, n=num_tweets)


# In[19]:


all_gov_tweets_df = pd.DataFrame(user_tweets)
all_gov_tweets_df.to_csv(f"data/gov_tweets_{num_tweets}.csv", sep=',', index=False)


# ### Exclude tweets before 1st April 2020

# In[22]:


pd.to_datetime(all_gov_tweets_df['date'])


# In[23]:


start_datetime = pd.to_datetime(start_date)
gov_tweets_df = all_gov_tweets_df[pd.to_datetime(all_gov_tweets_df['date']) >= start_datetime.tz_localize('UCT')]


# In[24]:


gov_tweets_df


# ### Check counts of each user

# In[25]:


user_counts = {}
for t_user in target_users:
    user_counts[t_user] = len(gov_tweets_df[gov_tweets_df['user'] == t_user])
user_counts


# ## Sentiment Analyzer

# In[26]:


def run_downloads():
    nltk.download('twitter_samples')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    
# run_downloads()
    
stop_words = stopwords.words('english')


# In[27]:


def tokenize(tweet_tokens):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"                       "(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", '', token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def remove_noise(tweet_tokens):
    all_cleaned_tokens = []
    for tweet_tokens in tweet_tokens:
        all_cleaned_tokens.append(tokenize(tweet_tokens))
    return all_cleaned_tokens

def get_all_words(cleaned_tokens):
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token
            
def get_dataset_from_tokens(cleaned_tokens, tag):
    dataset = []
    for tweet_tokens in cleaned_tokens:
        bag_of_tokens = {token: True for token in tweet_tokens}
        dataset.append((bag_of_tokens, tag))
    return dataset


# In[28]:


# Get training data
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Get tokenized training data
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')


# In[29]:


# Remove noise (normalize + stop word removal)
positive_cleaned_tokens = remove_noise(positive_tweet_tokens)
negative_cleaned_tokens = remove_noise(negative_tweet_tokens)


# In[30]:


# Word frequency distributions
positive_freq_dist = FreqDist(get_all_words(positive_cleaned_tokens))
negative_freq_dist = FreqDist(get_all_words(negative_cleaned_tokens))
print(positive_freq_dist.most_common(10))
print(negative_freq_dist.most_common(10))


# In[31]:


# Convert data to NLTK-required format
positive_dataset = get_dataset_from_tokens(positive_cleaned_tokens, "Positive")
negative_dataset = get_dataset_from_tokens(negative_cleaned_tokens, "Negative")
dataset = positive_dataset + negative_dataset


# In[32]:


# Split data
split_ratio = 0.7
split = int(len(dataset) * split_ratio)
random.shuffle(dataset)
train_data = dataset[slice(0, split)]
test_data = dataset[slice(split, len(dataset))]


# In[33]:


# Build model
classifier = NaiveBayesClassifier.train(train_data)
print(f"Accuracy is: {classify.accuracy(classifier, test_data)}")
print(classifier.show_most_informative_features(10))


# ### Insert sentiment predictions

# In[34]:


def classify_tweet(tweet_text):
    tweet_tokens = set(word.lower() for word in word_tokenize(tweet_text))
    cleaned_tweet_tokens = remove_noise([tweet_tokens])
    bag_of_tokens = {token: True for token in cleaned_tweet_tokens[0]}
    prediction = classifier.classify(bag_of_tokens)
    return prediction


# In[35]:


classify_tweet('hello, how dog cat beautiful')


# In[36]:


final_tweet_df = gov_tweets_df
final_tweet_df['sentiment'] = [classify_tweet(text) for text in final_tweet_df['text']]


# In[37]:


final_tweet_df


# In[38]:


pos_neg_counts = {}
for t_user in target_users:
    pos_neg_counts[t_user] = {'Positive': len(gov_tweets_df[(gov_tweets_df['sentiment'] == 'Positive') &
                                                            (gov_tweets_df['user'] == t_user)]),
                              'Negative': len(gov_tweets_df[(gov_tweets_df['sentiment'] == 'Negative') &
                                                            (gov_tweets_df['user'] == t_user)])}
pos_neg_counts


# ### Sort tweets by day

# In[70]:


date_range = [datetime.datetime(2020, month, day) for month in range(4, 7) for day in range(1,31)]
date_range.append(datetime.datetime(2020, 5, 31))
date_range.sort()


# In[66]:


(datetime.datetime(pd.to_datetime(row['date']).year, pd.to_datetime(row['date']).month, pd.to_datetime(row['date']).day))


# In[309]:


users_daily_sentiments = {date: collections.defaultdict(list) for date in date_range}

for index, row in final_tweet_df.iterrows():
    users_daily_sentiments[(datetime.datetime(pd.to_datetime(row['date']).year, 
                                              pd.to_datetime(row['date']).month, 
                                              pd.to_datetime(row['date']).day))][row['user']].append(row['sentiment'])


# In[310]:


users_daily_sentiments_count = {date: collections.defaultdict(list) for date in date_range}

for date in users_daily_sentiments:
    for t_user in users_daily_sentiments[date]:
        i_pos = [s == 'Positive' for s in users_daily_sentiments[date][t_user]]
        i_neg = [s == 'Negative' for s in users_daily_sentiments[date][t_user]]       
        users_daily_sentiments_count[date][t_user] = [sum(i_pos), sum(i_neg)]


# In[165]:


total_daily_sentiments_count = {date: {} for date in date_range}

for date in users_daily_sentiments_count:
    pos = sum([users_daily_sentiments_count[date][u][0] for u in users_daily_sentiments_count[date]])
    neg = sum([users_daily_sentiments_count[date][u][1] for u in users_daily_sentiments_count[date]])
    total_daily_sentiments_count[date] = [pos, neg]


# In[287]:


positive_daily_total = {k: v[0] for k, v in total_daily_sentiments_count.items()}
negative_daily_total = {k: v[1] for k, v in total_daily_sentiments_count.items()}


# ## Plot Graphs

# (I'm running out of time so I have left the non-essential graphs a bit messy)

# In[300]:


if not os.path.exists('graphs'):
    os.makedirs('graphs')


# In[180]:


plt.hist(positive_daily_total)
plt.hist(negative_daily_total)


# In[260]:


for state in target_users.values():
    plt.plot(US_df[state], label=state)
    plt.legend(loc='right')


# In[303]:


i = 0
for state in target_users.values():
    plt.figure(i)
    i += 1
    plt.title(f"Daily Increase in Confirmed COVID-19 Cases is {state}")
    plt.plot(increase_df[state], label=state)
    plt.xlabel('datetime')
    plt.ylabel('number of tweets')
    
    ticks = (increase_df[state].keys())[::7]  
    plt.xticks(ticks, rotation = 45)
    
    plt.savefig(f"graphs/{state}_daily_increase_cases.png")


# From these plots, we can see that all the states seem to have a peak. However, this peak could have occured in April (New York, Rhode Island), May (Delaware, Indiana, Kansas, Maryland) or June (California, Georgia, Nevada, Texas).

# In[306]:


for t_user in target_users:
    u_pos = []
    u_neg = []
    dates = []
    for date in users_daily_sentiments_count:
        user_values = users_daily_sentiments_count[date][t_user]
        if len(user_values) > 0:
            u_pos.append(user_values[0])
            u_neg.append(user_values[1])
            dates.append(date)
            
    plt.figure(i)
    i += 1
    plt.scatter(dates, u_pos, label='Positive')
    plt.scatter(dates, u_neg, label='Negative')
    plt.legend()
    plt.title(f"Number of Positive and Negative Tweets from {target_users[t_user]}")
    plt.xlabel('datetime')
    plt.ylabel('number of tweets')
    
    ticks = dates[::7]  
    plt.xticks(ticks, rotation = 45)
    
    plt.savefig(f"graphs/{target_users[t_user]}_positive_negative_tweets.png")    


# From the above grpahs, you can see that only Clalifornia, Delaware, New York and Utah are tweeting consistently and with high numbers.

# In[308]:


datafull_states = ['California', 'Delaware', 'New York', 'Utah']


# ## Conclusion

# After initial investigation, it appears that there doesn't seem to be any correlation between the number of positive, negative or total tweets from state-level government officals and the spread of COVID-19.
# 
# A few reasons why this might be the case are:
# * The binary "Positive"/"Negative" classification was too simplistic to represent the affects of communication on the spread of the virus. A more suffisticated attempt could have included several different classifications: "Informative", "Questions", "Humorous", "Sarcastic" etc. Similarly, confidence weightings could have been applied to the predictions to get a better idea of how positive/negative tweets on a given day were.
# * The number of followers that a user has does not equate to the engagment of those followers. It is possible that only a few people saw and took notice of the officials' tweets. This could have been controlled for using a measurment of influence such as the T-Index. Data could also have been collected from the general public to better reflect actual conversations (however, given the time and monetary constraints this was not feasable for this initial project).
# * The effects of the tweets may have a delay of several days, or even weeks. For example, someone may see a tweet telling them to buy a mask, but not buy one for another few days. This could have been confounded further by the early asymptomatic nature of the virus. With more time and data, it would be possible to conducted lagged correlations to see if this is the case.

# ## Closing Remarks

# I would like to thank you for taking the time to read this notebook. I found it to be a very fun problem, even though the outcome was not as impactful as I'd hoped... at least, for the time being.
# 
# I also learnt a couple of new skills throughout The Data Incubator Challenge, and also got a good chance to practice a few skills that I hadn't used for a while.
# 
# Thanks again!
