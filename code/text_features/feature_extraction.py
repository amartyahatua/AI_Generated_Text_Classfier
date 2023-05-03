# Source: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
import pandas as pd
import string
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from readability import Readability
import spacy
import language_tool_python

nlp = spacy.load("en_core_web_sm")

pos_family = {
    'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
    'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj': ['JJ', 'JJR', 'JJS'],
    'adv': ['RB', 'RBR', 'RBS', 'WRB']
}


# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


def readbility_Score(trainDF):
    result = pd.DataFrame()
    for i in range(trainDF.shape[0]):
        try:
            temp_result = []
            r = Readability(trainDF['Text'].iloc[i])

            flesch_kincaid = r.flesch_kincaid()
            temp_result.append(flesch_kincaid.score)

            flesch = r.flesch()
            temp_result.append(flesch.score)

            gunning_fog = r.gunning_fog()
            temp_result.append(gunning_fog.score)

            coleman_liau = r.coleman_liau()
            temp_result.append(coleman_liau.score)

            dale_chall = r.dale_chall()
            temp_result.append(dale_chall.score)

            ari = r.ari()
            temp_result.append(ari.score)

            linsear_write = r.linsear_write()
            temp_result.append(linsear_write.score)

            spache = r.spache()
            temp_result.append(spache.score)

            temp_result = pd.DataFrame([temp_result],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])
        except:
            temp_result = pd.DataFrame([[0] * 8],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])

        result = pd.concat([result, temp_result], axis=0)
    result = result.reset_index(drop=True)
    return result


def count_ent(trainDF):
    ner_count = []
    for i in range(trainDF.shape[0]):
        doc = nlp(trainDF['Text'].iloc[i])
        ner = []
        if doc.ents:
            for ent in doc.ents:
                ner.append(ent.text)
        else:
            print('NO EN')
        ner_count.append(len(ner))
    ner_count = pd.DataFrame(ner_count, columns=['NER Count'])
    return ner_count


def count_grammar_error(df):
    x = tool.check(df)
    return len(x)


trainDF = pd.read_csv('../../data/chatgpt_generated_wiki_data_1_5000.csv')
trainDF = trainDF.iloc[0:10, :]

# Removing rows having texts less than 100 words
trainDF['Text'].replace('', np.nan, inplace=True)
trainDF.dropna(subset=['Text'], inplace=True)
trainDF["word_count"] = trainDF["Text"].apply(lambda x: len(x))
trainDF = trainDF[(trainDF["word_count"] >= 100)]

# Text,GPT_Generated_Text
trainDF['char_count'] = trainDF['Text'].apply(len)
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
trainDF['punctuation_count'] = trainDF['Text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
trainDF['title_word_count'] = trainDF['Text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['Text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

trainDF['noun_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'pron'))

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['Text'])
vectorizer_count = count_vect.transform(trainDF['Text'])
vectorizer_count = pd.DataFrame(vectorizer_count.toarray())

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['Text'])
xtrain_tfidf = tfidf_vect.transform(trainDF['Text'])
tfidf_word = pd.DataFrame(xtrain_tfidf.toarray())

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['Text'])
trainDF_tfidf_ngram = tfidf_vect_ngram.transform(trainDF['Text'])
tfidf_ngram = pd.DataFrame(trainDF_tfidf_ngram.toarray())
#
# # characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                         max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['Text'])
trainDF_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF)
tfidf_ngram_chars = pd.DataFrame(trainDF_tfidf_ngram_chars.toarray())

# Get readability score
readbility = readbility_Score(trainDF)

# Get NER count
nercount = count_ent(trainDF)

feature_set = pd.concat([trainDF, vectorizer_count, tfidf_word, tfidf_ngram, tfidf_ngram_chars, readbility, nercount],
                        axis=1)
print(feature_set.shape)
feature_set.to_csv('../../data/features.csv')

# Grammar check
tool = language_tool_python.LanguageTool('en-US')
trainDF['text_error_length'] = trainDF['Text'].apply(count_grammar_error)
