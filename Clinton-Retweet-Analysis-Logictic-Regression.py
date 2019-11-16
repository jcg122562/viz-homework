import string

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
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
         'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count','hour', 'session','hashtag_count',
         'mention_count', 'url_count', 'work_count', 'stronger_count']]


# these are the columns that I will use for SKLEARN
df_retweet = df[['retweet_class', 'hashtag_count', 'mention_count', 'week_day', 'hour', 'session',
                 'url_count', 'favorite_count', 'work_count', 'stronger_count']]

# the are the columns / features that I will use for the test

feature_names = ['favorite_count', 'hashtag_count', 'mention_count', 'week_day', 'hour', 'url_count', 'session']

# Separating features(X) and target(y) for retweet
X = df_retweet.drop('retweet_class', axis=1)
y = df_retweet['retweet_class']

# Splitting features and target datasets into: train and test -- I will test it for .35 .45 .55
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)


# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))
cnf_matrix = (confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

# Heat Map
sns.set()

class_names=[1,2,3,4] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="autumn", fmt='g', center=0.00)

ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
plt.tight_layout()

plt.show()
# end of script

