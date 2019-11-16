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

# add actual time
df['actual_time'] = df['time'].str[11:]
df['actual_time'] = pd.to_datetime(df["actual_time"], format='%H:%M:%S').dt.time


def hr_func(ts):
    return ts.hour

# get the hour
df['hour'] = df['actual_time'].apply(hr_func)
df['hour'].astype(int)

# get the session of the hour
df['session'] = np.where(df['hour'] <= 4, 1,
                np.where(df['hour'].between(5, 12), 2,
                np.where(df['hour'].between(13, 18), 3, 4)))

# Get Clinton only
df = df[df['handle'] == 'HillaryClinton']
# max retweet_count to 50000 anything outside there is irregular -- to get the realistic tweet this still draws 3210
# records out of 3226
df = df[df['retweet_count'] < 12500]
# print(df.shape)
# Get month year and month
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


df1 = df[['retweet_class', 'retweet_count']]




# add post_count
df['post_count'] = df['handle'].count()

# get weekday
df['week_day'] = df['actual_date'].dt.dayofweek

df['hashtag_count'] = df.text.str.count("#")
df['mention_count'] = df.text.str.count("@")
df['url_count'] = df.text.str.count("https")


print(f'url count:', df['url_count'].count())


df = df[['retweet_class', 'handle', 'text', 'time', 'actual_date', 'actual_time', 'week_day',
         'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count','hour', 'session','hashtag_count',
         'mention_count', 'url_count']]


# for sklearn columns
df_retweet = df[['retweet_class', 'hashtag_count', 'mention_count', 'week_day', 'hour', 'session',
                 'url_count', 'favorite_count']]

# column_names

feature_names = ['hashtag_count', 'mention_count', 'week_day', 'hour', 'session', 'url_count', 'favorite_count']

# Separating features(X) and target(y) for retweet
X = df_retweet.drop('retweet_class', axis=1)
y = df_retweet['retweet_class']

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# # Printing original Dataset
# print(f"X.shape: {X.shape}, y.shape: {y.shape}")


# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, feature_names)}")

# print(f'Dataset X shape: {X.shape}')
# print(f'Dataset y shape: {y.shape}')

# Predicting the results for our test dataset
predicted_values = lm.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")


sns.set(palette="pastel")

# Plotting differenct between real and predicted values
sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

# Plotting the residuals: the error between the real and predicted values
residuals = y_test - predicted_values
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plt.show()

# Understanding the error that we want to minimize
from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")

