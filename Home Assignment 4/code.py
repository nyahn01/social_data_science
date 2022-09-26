#!/usr/bin/env python3
# ------------------------- CODE CELL injected: setup --------------------------
__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOT_REGISTRY__ = []
__LABEL__ = None


# If matplotlib is available on the test system, it's set to headless mode and all plots are stored
# on disk rather than displayed.
try:
    print("use 'Agg' backend for matplotlib")
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt

    # store current matplotlib figure to disk
    def _dump_figure():
        from pathlib import Path
        from autograde.util import snake_case

        # check if something was plotted
        if plt.gcf().get_axes():
            global __LABEL__
            global __PLOT_REGISTRY__

            # ensure plots folder exists
            root = Path('figures')
            if not root.is_dir():
                root.mkdir()

            # infer name
            name = snake_case(f'fig_{__LABEL__}_{len(__PLOT_REGISTRY__) + 1}')
            path = root.joinpath(name)
            __PLOT_REGISTRY__.append(path)

            # store current figure
            print(f'save current figure: {path}')
            plt.savefig(path)
            plt.close()

    plt.show = lambda *args, **kwargs: _dump_figure()

except ImportError:
    pass


dump_figure = globals().get('_dump_figure', lambda *args, **kwargs: None)


# inform user about import filters used
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')

# EXECUTED IN 0.103s
# STDOUT
#     use 'Agg' backend for matplotlib


# ------------------------------- CODE CELL nb-1 -------------------------------
# credentials of all team members (you may add or remove items from the list)
team_members = [
    {
        'first_name': 'Na Young',
        'last_name': 'Ahn',
        'student_id': 392326
    },
    {
        'first_name': 'Esther',
        'last_name': 'Tala',
        'student_id': 368095
    },
    {
        'first_name': 'Mika',
        'last_name': 'Rosin',
        'student_id': 395049
    },
    {
        'first_name': 'Laurin',
        'last_name': 'Ellenbeck',
        'student_id': 372280
    }
]

# injected by test
dump_figure()

# EXECUTED IN 0.00153s


# ------------------------------- CODE CELL nb-2 -------------------------------
# general imports may go here
import nltk
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# injected by test
dump_figure()

# EXECUTED IN 0.000394s


# ------------------------------- CODE CELL nb-3 -------------------------------
df_stance = pd.read_csv("tweets_annotated.csv", index_col = 0)
df_stance.head()

# injected by test
dump_figure()

# EXECUTED IN 0.00904s


# ------------------------------- CODE CELL nb-4 -------------------------------
# stance
df_stance_grouped_stance = df_stance.groupby('Target').Stance.value_counts(normalize=False).unstack().T
df_stance_grouped_sentiment = df_stance.groupby('Target').Sentiment.value_counts(normalize=False).unstack().T

num_items= len(df_stance_grouped_sentiment.columns)
#fig, ax = plt.subplots( nrows= 2, ncols=num_items, figsize = (22,8), num=1)
df_stance_grouped_stance = df_stance_grouped_stance.T
df_stance_grouped_sentiment = df_stance_grouped_sentiment.T

i = 0 

fig, ax = plt.subplots( nrows= 1, ncols=num_items, figsize = (25,12))
for stance, sentiment in zip(df_stance_grouped_stance.iterrows(), df_stance_grouped_sentiment.iterrows()):
    
    stance = pd.DataFrame(stance[1]).reset_index()
    theme = stance.columns[1]
    stance = stance.rename(columns={stance.columns[1]: "stance", 'Stance': 'class'})
    sentiment = pd.DataFrame(sentiment[1]).reset_index()
    sentiment=sentiment.rename(columns={'Sentiment':'class', sentiment.columns[1] : 'sentiment'})
    concat = stance.set_index('class').join(sentiment.set_index('class'), on='class').rename_axis(theme).rename({-1:'negative', 0:'neutral', 1:'positive'})
    concat.plot.bar(ax=ax[i]) 
    i+=1
plt.savefig("stance_sentiment.png")
plt.show()

# injected by test
dump_figure()

# EXECUTED IN 2.29s
# STDOUT
#     save current figure: figures/fig_nb_4_1


# ------------------------------- CODE CELL nb-5 -------------------------------
Stance = df_stance.Stance
Sentiment = df_stance.Sentiment
xtab_stance = pd.crosstab(Stance, Sentiment)
xtab_stance

# injected by test
dump_figure()

# EXECUTED IN 0.0243s


# ------------------------------- CODE CELL nb-6 -------------------------------
# download vader_lexicon if you haven't done so already
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# prepare input data
tweets = df_stance.Tweet
results = []
length = len(df_stance.index)
for i in range(length):
    score = analyzer.polarity_scores(df_stance.Tweet.iloc[i])['compound']
    if score < -0.1: 
        results.append(-1)
    elif score > 0.1: 
        results.append(1)
    else:
        results.append(0)

results = pd.DataFrame(results).rename(columns={0: 'classes'})

# injected by test
dump_figure()

# EXECUTED IN 1.32s


# ------------------------------- CODE CELL nb-7 -------------------------------
y_vader = results

# injected by test
dump_figure()

# EXECUTED IN 0.000387s


# ------------------------------- CODE CELL nb-8 -------------------------------
cm_stance = confusion_matrix( df_stance.Stance.to_numpy(),results.to_numpy().squeeze(), labels=[-1, 0, 1])
acc_stance = accuracy_score(y_true=df_stance.Stance.to_numpy(), y_pred=results.to_numpy().squeeze(), normalize=True)
print ('acc_stance = ', acc_stance, '\n', 'cm_stance: \n ', cm_stance)

# injected by test
dump_figure()

# EXECUTED IN 0.00393s
# STDOUT
#     acc_stance =  0.3772782503037667 
#      cm_stance: 
#       [[329 264 357]
#      [  5   3   1]
#      [207 191 289]]


# ------------------------------- CODE CELL nb-9 -------------------------------
cm_sentiment = confusion_matrix( df_stance.Sentiment.to_numpy(),results.to_numpy().squeeze(), labels=[-1, 0, 1])
acc_sentiment =  accuracy_score(y_true=df_stance.Sentiment.to_numpy(), y_pred=results.to_numpy().squeeze(), normalize=True)
print ('acc_sentiment = ', acc_sentiment, '\n', 'cm_sentiment: \n ', cm_sentiment)

# injected by test
dump_figure()

# EXECUTED IN 0.00381s
# STDOUT
#     acc_sentiment =  0.4975698663426488 
#      cm_sentiment: 
#       [[474 302 316]
#      [ 10  40  26]
#      [ 57 116 305]]


# ------------------------------ CODE CELL nb-10 -------------------------------
### read in data

# read in reviews
with open("epinions.txt") as f:
    reviews = f.readlines()
reviews = [x.strip() for x in reviews] 
f.close()

# read in ratings
ratings = np.loadtxt("epinions_ratings.txt", unpack=False)

# injected by test
dump_figure()

# EXECUTED IN 0.186s


# ------------------------------ CODE CELL nb-11 -------------------------------
# feel free to explore the reviews yourself
print(reviews[0])
print(ratings[0])

# injected by test
dump_figure()

# EXECUTED IN 0.0004s
# STDOUT
#     i got this printer from minolta as a warranty replacement for the minolta pagepro 1100l which didnt work under windows xp the unit is a little bulkier than the 1100l also the paper tray is fully enclosed however the plastic hinged top does not close as fully as the 1100l did and more dust gets into the printer
#     4.0


# ------------------------------ CODE CELL nb-12 -------------------------------
import string  

def review_tokenizer(review):
    t = review.lower() # lowercase
    t = [c for c in t if not (c in string.punctuation)] # remove punctuation
    t = ''.join(t) # convert back to string
    return t.strip().split() # tokenize

# injected by test
dump_figure()

# EXECUTED IN 0.000378s


# ------------------------------ CODE CELL nb-13 -------------------------------
word_count = dict()

for review in reviews:
    tokens = review_tokenizer(review)
    for token in tokens:
        if token not in word_count:
            word_count[token] = 1 
        else:
            word_count[token] += 1

# injected by test
dump_figure()

# EXECUTED IN 3.24s


# ------------------------------ CODE CELL nb-14 -------------------------------
print(word_count["i"])
print(word_count["printer"])
print(word_count["dust"])
print(word_count["minolta"])

# injected by test
dump_figure()

# EXECUTED IN 0.000407s
# STDOUT
#     39519
#     834
#     135
#     43


# ------------------------------ CODE CELL nb-15 -------------------------------
n_words = len(word_count)

# injected by test
dump_figure()

# EXECUTED IN 0.000324s


# ------------------------------ CODE CELL nb-16 -------------------------------
most_freq_words = sorted(word_count.items(), key=lambda item: item[1])[-20:]

# injected by test
dump_figure()

# EXECUTED IN 0.0284s


# ------------------------------ CODE CELL nb-17 -------------------------------
def build_feature_matrix(documents: List[str], dictionary: List[str], tokenizer = review_tokenizer) -> pd.DataFrame:
    """
    :param documents: list of strings, each string representing a document (in our case, a review)
    :param dictionary: list of strings, each string representing a word in our dictionary
    :param tokenizer: tokenizing function that has to be used to preprocess each document, 
    :                defaults to the given review_tokenizer
    :return: pandas dataframe of binary occurance scores - i-th row represents i-th document, 
    :        columns should be named after the word that they represent
    """
    
    rows = []
    
    for document in documents:
        tokens = review_tokenizer(document)
        row = [1 if i in tokens else 0 for i in dictionary]
        rows.append(row)
    
    return pd.DataFrame(rows, columns=dictionary)

# injected by test
dump_figure()

# EXECUTED IN 0.000481s


# ------------------------------ CODE CELL nb-18 -------------------------------
from sklearn.linear_model import Ridge

zip_1d, _ = zip(*sorted(word_count.items(), key=lambda item: item[1])[-1000:])
most_freq_1000_words = zip_1d
df_bin_matrix = build_feature_matrix(reviews, most_freq_1000_words, review_tokenizer)
df_bin_matrix.head()

# injected by test
dump_figure()

# EXECUTED IN 19.2s


# ------------------------------ CODE CELL nb-19 -------------------------------
reg_bin = Ridge(alpha=1.0)
reg_bin.fit(df_bin_matrix, ratings)
y_bin_pred = reg_bin.predict(df_bin_matrix)

# injected by test
dump_figure()

# EXECUTED IN 0.311s


# ------------------------------ CODE CELL nb-20 -------------------------------
r2_2d = r2_score(ratings, y_bin_pred)
print(r2_2d)

# injected by test
dump_figure()

# EXECUTED IN 0.00117s
# STDOUT
#     0.4060235406284728


# ------------------------------ CODE CELL nb-21 -------------------------------
all_2d = sorted(dict(zip(most_freq_1000_words, reg_bin.coef_.tolist())).items(), key=lambda item: item[1])

l_highest_2d = all_2d[-10:]
l_lowest_2d = all_2d[:10]

print(l_highest_2d)
print(l_lowest_2d)

# injected by test
dump_figure()

# EXECUTED IN 0.00124s
# STDOUT
#     [('world', 0.3348208984330826), ('carpet', 0.33936600806765305), ('bike', 0.3423876244777322), ('love', 0.36908983599548495), ('hyundai', 0.3745671653672038), ('excellent', 0.37576550325049984), ('amazing', 0.3974992246271069), ('britax', 0.481751738768184), ('rover', 0.643223772690684), ('mop', 0.8749934118190137)]
#     [('verizon', -0.7038085497514566), ('maytag', -0.6077990019317149), ('router', -0.5434296338400832), ('land', -0.5285409871217064), ('slow', -0.4941531455417726), ('stay', -0.48882287787929257), ('poor', -0.484503285031196), ('unfortunately', -0.40604874591791423), ('acura', -0.39049699631746926), ('oven', -0.35428562648642853)]


# ------------------------------ CODE CELL nb-22 -------------------------------
def build_tfidf_matrix(documents: List[str], dictionary: List[str], tokenizer = review_tokenizer) -> pd.DataFrame:
    """
    :param documents: list of strings, each string representing a document (in our case, a review)
    :param dictionary: list of strings, each string representing a word in our dictionary
    :param tokenizer: tokenizing function that has to be used to preprocess each document, 
    :                defaults to the given review_tokenizer
    :return: pandas dataframe of tf-idf scores - i-th row represents i-th document, 
    :        columns should be named after the word that they represent
    """
    
    rows = []
    idf = {}
    N = len(documents)
    
    tokenized = []
    
    for document in documents:
             tokenized.append(review_tokenizer(document))

    for word in dictionary:
        count = 0.0
        for i in range(N):
            if word in tokenized[i]:
                count += 1
        idf[word] = math.log(N / count)
    
    for tokens in tokenized:
        row = [(tokens.count(i) / float(len(tokens))) * idf[i] for i in dictionary]        
        
        rows.append(row)
        
    return pd.DataFrame(rows, columns=dictionary)

zip_1f, _ = zip(*sorted(word_count.items(), key=lambda item: item[1])[-10:])
most_freq_1000_words = zip_1f
df_tfidf_matrix = build_tfidf_matrix(reviews, most_freq_1000_words, review_tokenizer)
df_tfidf_matrix.head()

# injected by test
dump_figure()

# EXECUTED IN 2.77s


# ------------------------------ CODE CELL nb-23 -------------------------------
zip_1f, _ = zip(*sorted(word_count.items(), key=lambda item: item[1])[-1000:])
most_freq_1000_words = zip_1f
df_tfidf_matrix = build_tfidf_matrix(reviews, most_freq_1000_words, review_tokenizer)
df_tfidf_matrix.head()

# injected by test
dump_figure()

# EXECUTED IN 51.2s


# ------------------------------ CODE CELL nb-24 -------------------------------
reg_tfidf = Ridge(alpha=1.0)
reg_tfidf.fit(df_tfidf_matrix, ratings)
y_tfidf_pred = reg_bin.predict(df_tfidf_matrix)

# injected by test
dump_figure()

# EXECUTED IN 0.364s


# ------------------------------ CODE CELL nb-25 -------------------------------
r2_2f = r2_score(ratings, y_tfidf_pred)
print(r2_2f)

# injected by test
dump_figure()

# EXECUTED IN 0.00117s
# STDOUT
#     0.028337003262503213


# ------------------------------ CODE CELL nb-26 -------------------------------
df_sample = pd.read_csv("sample_dis.csv", sep=';', index_col=0)
df_sample

# injected by test
dump_figure()

# EXECUTED IN 0.00539s


# ------------------------------ CODE CELL nb-27 -------------------------------
print("Row Marginals:")
adult_marginals = pd.Series({'Child' : 11500, 
                        'Adult': 40150})
print(adult_marginals)


wealth_marginals = pd.Series({'Poor' : 30000, 
                         'Middle Class': 21350,
                         'Rich' : 300 })
print("\nColumn Marginals:")
print(wealth_marginals)

# injected by test
dump_figure()

# EXECUTED IN 0.0268s
# STDOUT
#     Row Marginals:
#     Child    11500
#     Adult    40150
#     dtype: int64
#     
#     Column Marginals:
#     Poor            30000
#     Middle Class    21350
#     Rich              300
#     dtype: int64


# ------------------------------ CODE CELL nb-28 -------------------------------
def rake(sample_frame: pd.DataFrame, row_marginals: pd.Series, col_marginals: pd.Series, eps: Optional[float] = 1e-03):
    """
    :param sample_frame: pandas dataframe which cross-tabulates frequencies of two attributes
    :param row_marginals: pandas series representing the marginal distribution of the row attribute
    :param col_marginals: pandas series representing the marginal distribution of the column attribute
    :param eps: float, convergence parameter
    :return: pandas dataframe of sample weights
    """
   
    
    # your code here
    init_scaling_factor = col_marginals.sum()/sample_frame.values.sum()
    sample_frame = init_scaling_factor * sample_frame
    
#    iter = 0
#    maxiter = 5
    error = 10000
    while (error > eps):
#    while (error > 1e-3) and (iter < maxiter):
#        iter = iter + 1
        row_scaling_factors = row_marginals / sample_frame.T.loc[:].sum()
        sample_frame = row_scaling_factors.to_numpy().reshape(2,1) * sample_frame

        col_scaling_factors = col_marginals / sample_frame.loc[:].sum()
        sample_frame = col_scaling_factors.to_numpy() * sample_frame
        
        error = sum(abs(sample_frame.T.loc[:].sum()- adult_marginals))
    
    return sample_frame/df_sample

# injected by test
dump_figure()

# EXECUTED IN 0.000473s


# ------------------------------ CODE CELL nb-29 -------------------------------
# true weights for given data, for you to test
df_weights = pd.read_csv("pstweights.csv", sep=';', index_col=0)
df_weights

# injected by test
dump_figure()

# EXECUTED IN 0.0324s


# ------------------------------ CODE CELL nb-30 -------------------------------
df_exercise = pd.read_csv("exercise.csv", sep=';', index_col=0)
df_exercise

# injected by test
dump_figure()

# EXECUTED IN 0.00531s


# ------------------------------ CODE CELL nb-31 -------------------------------
df_population = df_sample*df_weights
display(df_population)
total_population = sum(df_population.loc[:].sum())
df_total_hours = df_population * df_exercise
display(df_total_hours)
total_hours = sum(df_total_hours.loc[:].sum())

child_population = df_population.loc["Child"].sum()
child_hours = df_total_hours.loc["Child"].sum()

# injected by test
dump_figure()

# EXECUTED IN 0.149s
# STDOUT
#     Poor  Middle Class        Rich
#     Child   6687.56557   4759.317497   53.116933
#     Adult  23312.43443  16590.682503  246.883067
#                    Poor  Middle Class         Rich
#     Child  40125.393418  47593.174971   637.403199
#     Adult  46624.868861  99544.095017  2468.830668


# ------------------------------ CODE CELL nb-32 -------------------------------
av_total = total_hours/total_population
av_children = child_hours/child_population

display(av_total)
display(av_children)

# injected by test
dump_figure()

# EXECUTED IN 0.0028s
# STDOUT
#     4.58845626589662
#     7.683127964147604


# ------------------------ CODE CELL injected: teardown ------------------------
__IA_FLAG__ = True

# EXECUTED IN 0.000168s


