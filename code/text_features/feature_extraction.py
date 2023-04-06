# Source: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
import pandas as pd
import string
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from readability import Readability
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
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
            print('flesch_kincaid', flesch_kincaid.score)
            temp_result.append(flesch_kincaid.score)
            flesch = r.flesch()
            print('flesch', flesch.score)
            temp_result.append(flesch.score)

            gunning_fog = r.gunning_fog()
            print('gunning_fog', gunning_fog.score)
            temp_result.append(gunning_fog.score)

            coleman_liau = r.coleman_liau()
            print('coleman_liau', coleman_liau.score)
            temp_result.append(coleman_liau.score)

            dale_chall = r.dale_chall()
            print('dale_chall', dale_chall.score)
            temp_result.append(dale_chall.score)

            ari = r.ari()
            print('ari', ari.score)
            temp_result.append(ari.score)

            linsear_write = r.linsear_write()
            print('linsear_write', linsear_write.score)
            temp_result.append(linsear_write.score)

            spache = r.spache()
            print('spache', spache.score)
            temp_result.append(spache.score)
            temp_result = pd.DataFrame([temp_result],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])
        except:
            temp_result = pd.DataFrame([[0]*8],
                                       columns=['flesch kincaid score', 'flesch score', 'gunning fog score',
                                                'coleman liau score', \
                                                'dale chall score', 'ari score', 'linsear write score', 'spache score'])

        result = pd.concat([result, temp_result], axis=0)
    return result

def count_ent(trainDF):
    ner_count = pd.DataFrame()
    for i in range(trainDF.shape[0]):
        doc = nlp(trainDF['Text'].iloc[i])
        ner = []
        if doc.ents:
            for ent in doc.ents:
                ner.append(ent.text)
        else:
            print('NO EN')

    return ner_count

data = pd.read_csv('../../data/chatgpt_generated_wiki_data_1_5000.csv')
trainDF = data.iloc[0:10,:]
print(trainDF.shape)

# Text,GPT_Generated_Text
# trainDF['char_count'] = trainDF['Text'].apply(len)
# trainDF['word_count'] = trainDF['Text'].apply(lambda x: len(x.split()))
# trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
# trainDF['punctuation_count'] = trainDF['Text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# trainDF['title_word_count'] = trainDF['Text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# trainDF['upper_case_word_count'] = trainDF['Text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
#
# trainDF['noun_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'noun'))
# trainDF['verb_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'verb'))
# trainDF['adj_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'adj'))
# trainDF['adv_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'adv'))
# trainDF['pron_count'] = trainDF['Text'].apply(lambda x: check_pos_tag(x, 'pron'))
#print(trainDF)


# count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vect.fit(trainDF['Text'])
# vectorizer_count = count_vect.transform(trainDF['Text'])
# print(vectorizer_count.toarray())
#
# # create a count vectorizer object
# lda_model = LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
# x_topics = lda_model.fit_transform(vectorizer_count)
# topic_word = lda_model.components_
# print((topic_word[0].shape))


# # word level tf-idf
# tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
# tfidf_vect.fit(trainDF['text'])
# xtrain_tfidf =  tfidf_vect.transform(trainDF['Text'])
# print(xtrain_tfidf.toarray())
#
# # ngram level tf-idf
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram.fit(trainDF['Text'])
# trainDF_tfidf_ngram =  tfidf_vect_ngram.transform(trainDF['Text'])
# print(trainDF_tfidf_ngram.toarray())
#
# # characters level tf-idf
# tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram_chars.fit(trainDF['Text'])
# trainDF_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF)
# print(trainDF_tfidf_ngram_chars.toarray())


# Get readability score
#readbility_score = readbility_Score()

#print(trainDF['Text'].iloc[0])



count_ent(doc)
#print(len(ner))