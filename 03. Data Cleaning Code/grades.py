from sklearn.metrics.pairwise import cosine_similarity as cosim
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
import spacy
import scipy.stats as stats
import pandas as pd

CATEGORIES = ["food", "service"]

nltk.download('vader_lexicon')
sample_set = pd.read_csv("sample_set.csv")

if set(CATEGORIES) != set(sample_set.category):
    raise ValueError("Incorrect set of values provided")

def assignCategory(input_text):
    "Assigns provided text to a category."
    #TODO should change this to a better model at some point
    nlp = spacy.load('en_core_web_sm')
    max_cosim_index, max_cosim = -1, -1

    input_text_vector = nlp(input_text).vector
    for i, cat in enumerate(CATEGORIES):
        curr_cosim_val = cosim(input_text_vector.reshape(1, -1),
                                nlp(cat).vector.reshape(1, -1))
        if max_cosim < curr_cosim_val:
            max_cosim = curr_cosim_val
            max_cosim_index = i

    return CATEGORIES[max_cosim_index]

def score(input_text):
    sid = SentimentIntensityAnalyzer()
    sentiment_dict = sid.polarity_scores(input_text)
    return sentiment_dict["pos"]

def grade(review_text):
    cat = assignCategory(review_text)
    x_food = score(review_text)
    sample_cat = sample_set[sample_set["category"] == cat]
    percentile_val = stats.percentileofscore(sample_cat.score, float(x_food))
    if percentile_val < 20:
        return "F"
    elif percentile_val < 40:
        return "D"
    elif percentile_val < 60:
        return "C"
    elif percentile_val < 80:
        return "B"
    else:
        return "A"

mdf = pd.read_csv("merged.csv")

#test_text = mdf.loc[5000]["text"]
test_text = mdf.sample(1, random_state=54).iloc[0]["text"]
print(test_text)
print(assignCategory(test_text))
print(grade(test_text))