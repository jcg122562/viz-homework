# Tweet Analysis Project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_context('talk')


# uncomment these two lines so you can see all the records when you print or write instead of dot dot dot ...
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def pretty_print(name, to_print):
    print(f'{name} ')
    print(f'{to_print}\n\n')


# check if /data directory exists
data_dir_exists = os.path.isdir('./data')
if not data_dir_exists:
    os.mkdir('data')

# filename where to print the output for my reference
f = open("data/my_output_draft.txt", "w+")


def pretty_write(name, to_write):
    f.write(f'{name}:\r')
    f.write(f'{to_write}\n\n')


# read / load  the file
df = pd.read_csv(filepath_or_buffer='data/tweets.csv',
                 sep=',',
                 header=0)  # header starts in first line

# add actual date
df['actual_date'] = df['time'].str[:10]
df['actual_date'] = pd.to_datetime(df['actual_date'], format='%Y/%m/%d')

# add month year
df['month_year'] = df['actual_date'].dt.to_period('M')
df['month'] = df['actual_date'].dt.month
df['month'] = df['month'].astype(str)

# select only important columns
df = df[['handle', 'text', 'is_retweet', 'time',
         'actual_date', 'lang', 'retweet_count', 'favorite_count', 'month_year', 'month']]

# # range data from april 2016 to september 2016
df = df[
    (df['actual_date'] >= '2016-4-1') & (df['actual_date'] <= '2016-9-30')]

df = df[['handle', 'actual_date', 'retweet_count', 'favorite_count', 'month_year', 'month']]

# set the x axis here
x = np.arange(len(df.month_year.unique()))

# group by
# df = df.groupby(['month_year']).agg({'retweet_count': 'sum', 'favorite_count': 'sum'})
df = df.groupby(['handle', 'month_year'], as_index=False).agg({'retweet_count': 'sum', 'favorite_count': 'sum'})

# at this point we have the desired dataset

# Tweets count comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

bar_width = .2
b1 = ax.bar(x, df.loc[df['handle'] == 'HillaryClinton', 'retweet_count'], width=bar_width, label="Clinton")
# Same thing, but offset the x by the width of the bar.
b2 = ax.bar(x + bar_width, df.loc[df['handle'] == 'realDonaldTrump', 'retweet_count'], width=bar_width, label="Trump")

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(df.month_year.unique())

# Add legend.
ax.legend()

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
ax.set_xlabel('Month Year', labelpad=10)
ax.set_ylabel('Retweet Count', labelpad=10)
ax.set_title('Retweet Comparison', pad=10)

fig.tight_layout()

plt.show()
plt.close()
