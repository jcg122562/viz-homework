# Tweet Analysis Project

from mpl_toolkits.mplot3d import axes3d
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
sns.set()


def pretty_print(name, to_print):
    print(f'{name} ')
    print(f'{to_print}\n\n')


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

# add month year
df['month_year'] = df['actual_date'].dt.to_period('M')
df['month'] = df['actual_date'].dt.month
df['month'] = df['month'].astype(str)

# add post count
df['post_count'] = df['handle'].count()

# select only important columns to become the main dataframe
df = df[['handle', 'text', 'is_retweet', 'time',
         'actual_date', 'lang', 'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count']]

# # range data from april 2016 to september 2016
df = df[
    (df['actual_date'] >= '2016-4-1') & (df['actual_date'] <= '2016-9-30')]

# select these columns
df = df[['handle', 'actual_date', 'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count']]

# df2 dataframe
df2 = df[['handle', 'actual_date', 'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count']]

# df7 dataframe
df7 = df[['handle', 'actual_date', 'retweet_count', 'favorite_count', 'month_year', 'month', 'post_count']]

# dataframe with all dates
df1 = df[['handle', 'actual_date', 'retweet_count', 'favorite_count', 'month_year', 'month']]
df1 = df1.groupby(['handle', 'actual_date'], as_index=False).agg({'retweet_count': 'sum', 'favorite_count': 'sum'})

# set the x axis here
x = np.arange(len(df.month_year.unique()))

# group by
df3 = df.groupby(['handle', 'actual_date'], as_index=False).agg({'retweet_count': 'sum', 'favorite_count': 'sum',
                                                                 'post_count': 'count'})
df3 = df3[(df3['handle'] == 'HillaryClinton')]

# df groupby
df = df.groupby(['handle', 'month_year'], as_index=False).agg({'retweet_count': 'sum', 'favorite_count': 'sum',
                                                               'post_count': 'count'})

# at this point we have the desired dataset

# Tweets count comparison
# this is an initial work as I progress, I will improve my charts
fig, ax = plt.subplots(2, 3, figsize=(12, 8), dpi=100)

bar_width = .2

# Chart 1 -- Bar Chart
b1 = ax[0][0].bar(x, df.loc[df['handle'] == 'HillaryClinton', 'retweet_count'], width=bar_width,
                  label="Clinton")
# Same thing, but offset the x by the width of the bar.
b2 = ax[0][0].bar(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'retweet_count'], width=bar_width,
                  label="Trump")

ax[0][0].set_xticks(x + bar_width / 2)
ax[0][0].set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[0][0].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax[0][0].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax[0][0].legend(prop={'size': 10})

# Axis styling.
ax[0][0].spines['top'].set_visible(False)
ax[0][0].spines['right'].set_visible(False)
ax[0][0].spines['left'].set_visible(False)
ax[0][0].spines['bottom'].set_color('#DDDDDD')
ax[0][0].tick_params(bottom=False, left=False)
ax[0][0].set_axisbelow(True)
ax[0][0].yaxis.grid(True, color='#EEEEEE')
ax[0][0].xaxis.grid(False)

# Add axis and chart labels.
ax[0][0].set_xlabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[0][0].set_ylabel('Retweet Count', labelpad=10, fontsize=10, fontweight='bold')
ax[0][0].set_title('Retweet Comparison', pad=10, fontsize=12, fontweight='bold')

# chart 2 - line chart / plot

b1 = ax[1][0].plot(x, df.loc[df['handle'] == 'HillaryClinton', 'favorite_count'], label="Clinton",
                   marker='o', color='mediumvioletred')
# Same thing, but offset the x by the width of the bar.
b2 = ax[1][0].plot(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'favorite_count'], label="Trump",
                   marker="o", color='y')

ax[1][0].set_xticks(x + bar_width / 2)
ax[1][0].set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[1][0].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax[1][0].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax[1][0].get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax[1][0].legend(prop={'size': 10})

# Add axis and chart labels.
ax[1][0].set_xlabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[1][0].set_ylabel('Favorite Count', labelpad=10, fontsize=10, fontweight='bold')
ax[1][0].set_title('Favorite Comparison', pad=10, fontsize=12, fontweight='bold')

# Chart 3 - Scatter
plt.style.use("ggplot")

b1 = ax[0][1].scatter(x, df.loc[df['handle'] == 'HillaryClinton', 'favorite_count'], label="Clinton", alpha=0.7)

# Same thing, but offset the x by the width of the bar.
b2 = ax[0][1].scatter(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'favorite_count'], label="Trump",
                      alpha=0.7)
ax[0][1].set_xticks(x + bar_width)
ax[0][1].set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[0][1].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax[0][1].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax[0][1].get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax[0][1].legend(prop={'size': 10})

# Add axis and chart labels.
ax[0][1].set_xlabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[0][1].set_ylabel('Favorite Count', labelpad=10, fontsize=10, fontweight='bold')
ax[0][1].set_title('Favorite Comparison', pad=10, fontsize=12, fontweight='bold')

# Chart 4


b1 = ax[1][1].plot(df.loc[df['handle'] == 'HillaryClinton', 'retweet_count'], x, label="Clinton",
                   marker='*', color='mediumvioletred', linestyle='--', linewidth=2)
# Same thing, but offset the x by the width of the bar.
b2 = ax[1][1].plot(df.loc[df['handle'] == 'realDonaldTrump', 'retweet_count'], x + bar_width, label="Trump",
                   marker="*", color='y', linestyle='--', linewidth=2)


ax[1][1].set_yticks(x + bar_width / 2)
ax[1][1].set_yticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[1][1].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax[1][1].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax[1][1].get_xaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax[1][1].legend(prop={'size': 10})

# Add axis and chart labels.
ax[1][1].set_xlabel('Favorite Count', labelpad=10, fontsize=10, fontweight='bold')
ax[1][1].set_ylabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[1][1].set_title('Favorite Comparison', pad=10, fontsize=12, fontweight='bold')

# # chart 5


b1 = ax[1][2].bar(x, df.loc[df['handle'] == 'HillaryClinton', 'post_count'], width=bar_width,
                  label="Clinton",
                  color='orange')
# Same thing, but offset the x by the width of the bar.
b2 = ax[1][2].bar(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'post_count'], width=bar_width,
                  label="Trump",
                  color='brown')

ax[1][2].set_xticks(x + bar_width / 2)
ax[1][2].set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[1][2].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax[1][2].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax[1][2].legend(prop={'size': 10})

# Axis styling.
ax[1][2].spines['top'].set_visible(False)
ax[1][2].spines['right'].set_visible(False)
ax[1][2].spines['left'].set_visible(False)
ax[1][2].spines['bottom'].set_color('#DDDDDD')
ax[1][2].tick_params(bottom=False, left=False)
ax[1][2].set_axisbelow(True)
ax[1][2].yaxis.grid(True, color='#EEEEEE')
ax[1][2].xaxis.grid(False)

# Add axis and chart labels.
ax[1][2].set_xlabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[1][2].set_ylabel('Post Count', labelpad=10, fontsize=10, fontweight='bold')
ax[1][2].set_title('Post Count Comparison', pad=10, fontsize=12, fontweight='bold')

# Chart 6 Scatter
plt.style.use("ggplot")

b1 = ax[0][2].scatter(x, df.loc[df['handle'] == 'HillaryClinton', 'post_count'],
                      label="Clinton",
                      alpha=0.7,
                      color='g')

# Same thing, but offset the x by the width of the bar.
b2 = ax[0][2].scatter(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'post_count'],
                      label="Trump",
                      alpha=0.7,
                      color='y')
ax[0][2].set_xticks(x + bar_width)
ax[0][2].set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax[0][2].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax[0][2].yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax[0][2].get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax[0][2].legend(prop={'size': 10})

# Add axis and chart labels.
ax[0][2].set_xlabel('Month Year', labelpad=10, fontsize=10, fontweight='bold')
ax[0][2].set_ylabel('Post Count', labelpad=10, fontsize=10, fontweight='bold')
ax[0][2].set_title('Post Count Comparison', pad=10, fontsize=12, fontweight='bold')


fig.tight_layout()
plt.savefig('plots/01_ClintonVsTrumpReTweetFavoriteComparison.png', dpi=300)

# plt.show()

# end of first set

# start of second set

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


z3 = np.zeros(6)
dx = np.ones(6)
dy = np.ones(6)
dz = [1, 2, 3, 4, 5, 6]


b1 = ax.bar3d(x, df.loc[df['handle'] == 'HillaryClinton', 'post_count'], z3, dx, dy, dz,
              color='g',
              label="Clinton",
              shade=True)
# Same thing, but offset the x by the width of the bar.
b2 = ax.bar3d(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'post_count'], z3, dx, dy, dz,
              color='y',
              label="Trump",
              shade=True)

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
#ax.legend()  ### this is where I am getting the edgecolor error -- seems know bug

# Add axis and chart labels.
ax.set_xlabel('Month Year', labelpad=10, fontsize=8)
ax.set_ylabel('Post Count', labelpad=10, fontsize=8)
ax.set_title('Post Count Comparison', pad=10, fontsize=8)

fig.tight_layout()
plt.savefig('plots/02_ClintonTrumpTweetPostComparison3DBar.png', dpi=300)
plt.close()


# # Set 3
x = np.arange(len(df3.actual_date.unique()))

fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
plt.style.use("ggplot")

b1 = ax.scatter(x, df3.loc[df3['handle'] == 'HillaryClinton', 'retweet_count'], label="Retweet", alpha=0.7)

# Same thing, but offset the x by the width of the bar.
b2 = ax.scatter(x + bar_width, df3.loc[df3['handle'] == 'HillaryClinton', 'favorite_count'], label="Favorite",
                      alpha=0.7)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(df3.actual_date.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 10})

# Add axis and chart labels.
ax.set_xlabel('Tweet Date', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Favorite Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Favorite Comparison', pad=10, fontsize=12, fontweight='bold')

fig.tight_layout()
plt.savefig('plots/03_HillaryRetweetFavoriteScatter.png', dpi=300)
plt.close()


# Hillary's Tweets Only

x = np.arange(len(df.month_year.unique()))
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

ax1.plot(x, df.loc[df['handle'] == 'HillaryClinton', 'favorite_count'], label="Favorite",
         marker='o', color='red')
ax2.plot(x + bar_width, df.loc[df['handle'] == 'HillaryClinton', 'retweet_count'], label="Retweet",
         marker="o",
         color='g')

ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(df.month_year.unique())

ax1.legend()
ax1.set_title("Hillary Clinton Tweet Analysis")

ax1.set_ylabel("Count")

ax2.legend()
ax2.set_xlabel("Months")

fig.tight_layout()
plt.savefig('plots/04_hillarytweetsanalysis.png', dpi=300)
plt.close()


# Hillary's retweet and favorite scatter
x = np.arange(len(df.month_year.unique()))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.style.use("ggplot")

b1 = ax.scatter(x, df.loc[df['handle'] == 'HillaryClinton', 'retweet_count'], label="Retweet", alpha=0.7)
b2 = ax.scatter(x, df.loc[df['handle'] == 'HillaryClinton', 'favorite_count'], label="Favorite", alpha=0.7)

# b2 = ax.scatter(x + bar_width, df.loc[df['handle'] == 'HillaryClinton', 'favorite_count'], label="Favorite",
#                 alpha=0.7)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 10})

# Add axis and chart labels.
ax.set_xlabel('Tweet Date', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Clinton Retweet and Favorite', pad=10, fontsize=12, fontweight='bold')

fig.tight_layout()

plt.savefig('plots/05_HillaryRetweetFavoriteScatter3D.png', dpi=300)
plt.close()

# Trumps Tweet Only

x = np.arange(len(df.month_year.unique()))
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

ax1.plot(x, df.loc[df['handle'] == 'realDonaldTrump', 'favorite_count'], label="Favorite",
         marker='o', color='red')
ax2.plot(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'retweet_count'], label="Retweet",
         marker="o",
         color='g')

ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(df.month_year.unique())

ax1.legend()
ax1.set_title("Donald Trump Tweet Analysis")

ax1.set_ylabel("Count")

ax2.legend()
ax2.set_xlabel("Months")
fig.tight_layout()

plt.savefig('plots/06_Donaldtweetsanalysis.png', dpi=300)
plt.close()


# Trump's retweet and favorite scatter
x = np.arange(len(df.month_year.unique()))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.style.use("ggplot")

b1 = ax.scatter(x, df.loc[df['handle'] == 'realDonaldTrump', 'retweet_count'], label="Retweet", alpha=0.7)
b2 = ax.scatter(x, df.loc[df['handle'] == 'realDonaldTrump', 'favorite_count'], label="Favorite", alpha=0.7)


ax.set_xticks(x + bar_width)
ax.set_xticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 10})

# Add axis and chart labels.
ax.set_xlabel('Tweet Date', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Trump Retweet and Favorite', pad=10, fontsize=12, fontweight='bold')

fig.tight_layout()

plt.savefig('plots/07_DonaldRetweetFavoriteScatter3D.png', dpi=300)
plt.close()

# Heatmap


df5 = df[['handle', 'retweet_count', 'favorite_count', 'post_count']]

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df5.corr(), annot=True, cmap='winter')
ax.set_xticklabels(df5.columns, rotation=45)
ax.set_yticklabels(df5.columns, rotation=45)
plt.savefig('plots/08_TweetsHeatMap.png')

plt.close()

# Second File - Words Use

# read / load  the file
df = pd.read_csv(filepath_or_buffer='data/WordsUsed.csv',
                 sep=',',
                 header=0)  # header starts in first line
#add word desc
df['word_desc'] = np.where(df['rate']== 1, 'Positive', 'Negative')


# # filter clinton used words only
df1 = df[(df['handle'] == 'HillaryClinton')]
# for pie
df1clinton = df1.groupby(['handle', 'word_desc'], as_index=False).agg({'count': 'sum'})
# sort by count descending
df1 = df1.sort_values(by='count', ascending=False).groupby('rate').head(26)
df1 = df1.drop(df1[df1.word == 'trump'].index)
df1 = df1.drop(df1[df1.word == 'like'].index)


# # get the top 26 words

# filter Trump used words only
df2 = df[(df['handle'] == 'realDonaldTrump')]
df2Trump = df2.groupby(['handle', 'word_desc'], as_index=False).agg({'count': 'sum'})

df2 = df2.sort_values(by='count', ascending=False).groupby('rate').head(26)
df2 = df2.drop(df2[df2.word == 'trump'].index)
df2 = df2.drop(df2[df2.word == 'like'].index)


# positive words
df1pos = df1[df1['rate'] == 1]
df2pos = df2[df2['rate'] == 1]

# positive words
df1neg = df1[df1['rate'] == -1]
df2neg = df2[df2['rate'] == -1]




# Clinton Positive
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=100)

x = np.arange(len(df1pos.word.unique()))

bar_width = .2

# first chart bar

b1 = ax.barh(df1pos['word'], df1pos['count'], color='cornflowerblue', label="Clinton")
# # Same thing, but offset the x by the width of the bar.
# b2 = ax.barh(df1pos['word'], df1pos['count'],  label="Clinton")

# b2 = ax[0][0].barh(df2pos.loc['count'], x, width=bar_width,label="Trump")

ax.set_yticks(x + bar_width / 2)
# ax[0][0].set_yticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax.legend(prop={'size': 10})

# Axis styling.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add axis and chart labels.
ax.set_xlabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Words', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Positive Words', pad=10, fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/09_ClintonPositiveWords.png')

plt.close()

# Clinton Negative
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=100)

x = np.arange(len(df1neg.word.unique()))

bar_width = .2

# first chart bar

b1 = ax.barh(df1neg['word'], df1neg['count'], color='indianred', label="Clinton")

ax.set_yticks(x + bar_width / 2)
# ax[0][0].set_yticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax.legend(prop={'size': 10})

# Axis styling.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add axis and chart labels.
ax.set_xlabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Words', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Negative Words', pad=10, fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/10_ClintonNegativeWords.png')

plt.close()

# Trump Positive
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=100)

x = np.arange(len(df2pos.word.unique()))

bar_width = .2

# first chart bar

b1 = ax.barh(df2pos['word'], df2pos['count'], color='cornflowerblue', label="Trump")
ax.set_yticks(x + bar_width / 2)
# ax[0][0].set_yticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax.legend(prop={'size': 10})

# Axis styling.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add axis and chart labels.
ax.set_xlabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Words', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Positive Words', pad=10, fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/11_TrumpPositiveWords.png')

plt.close()

# Trump Negative
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=100)

x = np.arange(len(df2neg.word.unique()))

bar_width = .2

b1 = ax.barh(df2neg['word'], df2neg['count'], color='indianred', label="Trump")

ax.set_yticks(x + bar_width / 2)
# ax[0][0].set_yticklabels(df.month_year.unique())

# set the ticks font to 8
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)

# Add legend.
ax.legend(prop={'size': 10})

# Axis styling.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add axis and chart labels.
ax.set_xlabel('Count', labelpad=10, fontsize=10, fontweight='bold')
ax.set_ylabel('Words', labelpad=10, fontsize=10, fontweight='bold')
ax.set_title('Negative Words', pad=10, fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/12_TrumpNegativeWords.png')

plt.close()

# Word Cloud
# Clinton Positive

content = ''
for val in df1pos['word']:
    content = content + val + ' '

stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(content)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('plots/13_ClintonPositiveWordCloud.png')
plt.close()

# Clinton Negative
content = ''
for val in df1neg.word:
    content = content + val + ' '


stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(content)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('plots/14_ClintonNegativeWordCloud.png')
plt.close()


# Trump Positive

content = ''
for val in df2pos.word:
    content = content + val + ' '

stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(content)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('plots/15_TrumpPositiveWordCloud.png')
plt.close()

# Trump Negative
content = ''
for val in df2neg.word:
    content = content + val + ' '

stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(content)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('plots/16_TrumpNegativeWordCloud.png')
plt.close()

#  scatter retweet and favorite
# x = np.arange(len(df3.favorite_count))

fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
plt.style.use("ggplot")
plt.xlim(0, 50000)
plt.ylim(0, 100000)

b1 = ax.scatter(df7.loc[df7['handle'] == 'HillaryClinton', 'retweet_count'],
                df7.loc[df7['handle'] == 'HillaryClinton', 'favorite_count'], color="blue", label="Clinton", alpha=0.7)

# Same thing, but offset the x by the width of the bar.
b2 = ax.scatter(df7.loc[df7['handle'] == 'realDonaldTrump', 'retweet_count'],
                df7.loc[df7['handle'] == 'realDonaldTrump', 'favorite_count'], color="red", label="Trump", alpha=0.7)


# ensure the format of the y axis using the actual values and not exponential
ax.get_yaxis().get_major_formatter().set_scientific(False)

# Add legend.
ax.legend(prop={'size': 12})

# Add axis and chart labels.
ax.set_xlabel('Retweets', labelpad=10, fontsize=12, fontweight='bold')
ax.set_ylabel('Favorites', labelpad=10, fontsize=12, fontweight='bold')
ax.set_title('Clinton Vs Trump - Tweets Comparison', pad=10, fontsize=14, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/17_ScatterRetweetandFavorite.png')
plt.close()

# total positve and negative words


# Pie plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

ax1.pie(df1clinton['count'], labels=df1clinton['word_desc'] + ' ' + df1clinton['count'].astype(str), autopct='%1.1f%%')
ax1.set_title('Hillary Clinton', fontsize=10, fontweight='bold')

ax2.pie(df2Trump['count'], labels=df2Trump['word_desc']+ ' ' + df2Trump['count'].astype(str), autopct='%1.1f%%')
ax2.set_title('Donal Trump', fontsize=10, fontweight='bold')
ax1.set_xlabel('Positive and Negative Words Rate Comparison', fontsize=10, fontweight='bold')
fig.tight_layout()

plt.savefig('plots/18_PositiveNegativeWordsComparison.png', dpi=300)
plt.close()

