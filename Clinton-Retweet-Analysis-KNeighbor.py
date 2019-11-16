import string

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics

import datetime
import calendar

# check if /data directory exists
data_dir_exists = os.path.isdir('./data')
if not data_dir_exists:
    os.mkdir('data')

# read / load  the tweet dataset file
df = pd.read_csv(filepath_or_buffer='data/tweets.csv',
                 sep=',',
                 header=0)  # header starts in first line

# add actual date column
df['actual_date'] = df['time'].str[:10]
df['actual_date'] = pd.to_datetime(df['actual_date'], format='%Y/%m/%d')
df['actual_time'] = df['time'].str[11:]
df['actual_time'] = pd.to_datetime(df["actual_time"], format='%H:%M:%S').dt.time
# df['actual_time'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour

# Get Clinton only
df = df[df['handle'] == 'HillaryClinton']


# max retweet_count to 50000 anything outside there is irregular -- to get the realistic tweet this still draws 3210
# records out of 3226
# df = df[df['retweet_count'] < 12500]


def hr_func(ts):
    return ts.hour


df['hour'] = df['actual_time'].apply(hr_func)
df['hour'].astype(int)

# get the session of the hour
df['session'] = np.where(df['hour'] <= 4, 1,
                         np.where(df['hour'].between(5, 12), 2,
                                  np.where(df['hour'].between(13, 18), 3, 4)))

# get the month_year and month
df['month_year'] = df['actual_date'].dt.to_period('M')
df['month'] = df['actual_date'].dt.month
df['month'] = df['month'].astype(str)

# Get the retweet_class (TARGET) -- I will only have 4 classes.  The data as analyzed, anything above 5000 retweet
# should fall to the last class.
df['retweet_class'] = np.where(df['retweet_count'].between(1, 1000),
                               1,
                               np.where(df['retweet_count'].between(1001, 2000),
                                        2,
                                        np.where(df['retweet_count'].between(2001, 5000),
                                                 3,
                                                 4)))

# add post_count -- i am not using the post count as yet this will go in other analysis
df['post_count'] = df['handle'].count()

# get weekday -- i would like  to know if there is a pattern for high retweet based on what day it was posted
df['week_day'] = df['actual_date'].dt.dayofweek

# get the count of the #hashtag, @mention and URL  -- there is another way in getting this to be more exact however
# for this purposes, I will work on the easiest way to determine the count for each. In the next stage, I will attempt
# to
df['hashtag_count'] = df.text.str.count("#")
df['mention_count'] = df.text.str.count("@")
df['url_count'] = df.text.str.count("https")

# score the tweet based line is 20000  --- I will not use this one for now
# df['retweet_score'] = np.where(df['retweet_count'] == 1000,
#                                0,
#                       np.where(df['retweet_count'] < 1000,
#                                -1, 1))

# mention of word work
df['work_count'] = df.text.str.count("work")
df['stronger_count'] = df.text.str.count("stronger")

# these are the important columns of the dataframe
df = df[['retweet_class', 'handle', 'text', 'time', 'actual_date', 'actual_time', 'week_day',
         'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count', 'hour', 'session', 'hashtag_count',
         'mention_count', 'url_count', 'work_count', 'stronger_count']]

# these are the columns that I will use for SKLEARN
df_retweet = df[['retweet_class', 'hashtag_count', 'mention_count', 'week_day', 'hour', 'session',
                 'url_count', 'favorite_count', 'work_count', 'stronger_count']]

# the are the columns / features that I will use for the test

feature_names = ['favorite_count', 'hashtag_count', 'mention_count', 'week_day', 'hour', 'url_count', 'session']

# Separating features(X) and target(y) for retweet
# X = df_retweet.drop('retweet_class', axis=1)
X = np.array(df_retweet.drop('retweet_class', axis=1))
y = np.array(df_retweet['retweet_class'])

# Splitting features and target datasets into: train and test -- I will test it for .35 .45 .55
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)


clf = neighbors.KNeighborsClassifier(n_neighbors=5)
# clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
# example_measures = np.array([2, 2, 3, 1, 1, 3, 2, 2, 2])
# example_measures = example_measures.reshape(1, -1)
# prediction = clf.predict(example_measures)
# print(prediction)

#
#
# # Training a model using multiple differents algorithms and comparing the results
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import ElasticNet
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Lasso
#
# for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso]:
#     model = Model()
#     model.fit(X_train, y_train)
#     predicted_values = model.predict(X_test)
#     print(f"Printing RMSE error for {Model}: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")


# end of script
