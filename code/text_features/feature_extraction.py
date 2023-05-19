# Source: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
import pandas as pd
import string
import numpy as np
from textblob import TextBlob           #python -m pip install textblob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from readability import Readability     #pip install py-readability-metrics
                                        #python -m nltk.downloader punkt
import spacy
import language_tool_python             #python -m pip install language_tool_python
from octis_topicmodeling import train_save_neurallda

nlp = spacy.load("en_core_web_sm")      #python3 -m spacy download en_core_web_sm

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


def readbility_Score(trainDF, source=0):
    result = pd.DataFrame()
    for i in range(trainDF.shape[0]):
        try:
            temp_result = []
            r = Readability(trainDF['Text' if source==0 else 'GPT_Generated_Text'].iloc[i])

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


def count_ent(trainDF, source=0):
    ner_count = []
    for i in range(trainDF.shape[0]):
        doc = nlp(trainDF['Text' if source==0 else 'GPT_Generated_Text'].iloc[i])
        ner = []
        if doc.ents:
            for ent in doc.ents:
                ner.append(ent.text)
        #else:
        #    print('NO EN')
        ner_count.append(len(ner))
    ner_count = pd.DataFrame(ner_count, columns=['NER Count'])
    return ner_count


tool = language_tool_python.LanguageTool('en-US')
def count_grammar_error(df):
    x = tool.check(df)
    return len(x)

def feature_extraction(inputPath='./data/chatgpt_generated_wiki_data_1_5000.csv', source=0):
    if source == 1:
        selected_col = 'GPT_Generated_Text'
        lda_dataset_path = './data/wiki_5k_chatgpt'
        lda_model_folder = './model/lda_wiki_5k_chatgpt'
    else:
        selected_col = 'Text'
        lda_dataset_path = './data/wiki_5k_gt'
        lda_model_folder = './model/lda_wiki_5k_gt'
    trainDF = pd.read_csv(inputPath)
    #trainDF = trainDF.iloc[0:10, :]
    
    # Removing rows having texts less than 100 words
    trainDF[selected_col].replace('', np.nan, inplace=True)
    trainDF.dropna(subset=[selected_col], inplace=True)
    trainDF["char_count"] = trainDF[selected_col].apply(lambda x: len(x))
    trainDF = trainDF[(trainDF["char_count"] >= 100) & (~trainDF[selected_col].str.startswith('http://'))]
    print('step 1: ', trainDF.shape)

    # Text,GPT_Generated_Text
    trainDF['word_count'] = trainDF[selected_col].str.split().apply(len)
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
    trainDF['punctuation_count'] = trainDF[selected_col].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF[selected_col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    trainDF['noun_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'noun'))
    trainDF['verb_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'verb'))
    trainDF['adj_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adj'))
    trainDF['adv_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'adv'))
    trainDF['pron_count'] = trainDF[selected_col].apply(lambda x: check_pos_tag(x, 'pron'))
    print('step 2 char_count: ', len(trainDF['char_count']), len(trainDF['word_count']))

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF[selected_col])
    vectorizer_count = count_vect.transform(trainDF[selected_col])
    vectorizer_count = pd.DataFrame(vectorizer_count.toarray())
    print('step 3 vectorizer_count.shape: ', vectorizer_count.shape)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF[selected_col])
    xtrain_tfidf = tfidf_vect.transform(trainDF[selected_col])
    tfidf_word = pd.DataFrame(xtrain_tfidf.toarray())
    print('step 4 tfidf_word.shape: ', tfidf_word.shape)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF[selected_col])
    trainDF_tfidf_ngram = tfidf_vect_ngram.transform(trainDF[selected_col])
    tfidf_ngram = pd.DataFrame(trainDF_tfidf_ngram.toarray())
    print('step 5 tfidf_ngram.shape: ', tfidf_ngram.shape)
    #
    # # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                            max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF[selected_col])
    trainDF_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF)
    tfidf_ngram_chars = pd.DataFrame(trainDF_tfidf_ngram_chars.toarray())
    print('step 6 tfidf_ngram_chars.shape: ', tfidf_ngram_chars.shape)

    # Get readability score
    readbility = readbility_Score(trainDF, source)
    print('step 7 readbility.shape: ', readbility.shape)

    # Get NER count
    nercount = count_ent(trainDF, source)
    print('step 8 nercount.shape: ', nercount.shape)

    #extract topic modeling features (20 columns)
    _, _, lda_df = train_save_neurallda(dataset_path=lda_dataset_path, out_folder=lda_model_folder)

    #extract word features
    feat_df = trainDF[['word_count', 'char_count', 'word_density', 'punctuation_count', 'title_word_count', 'upper_case_word_count', \
                       'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]
    print('step 9 nercount.shape: ', feat_df.shape)
    #tfidf_ngram_chars has the shape (15, 5000) incompatible with the dataset => skipped
    feature_set = pd.concat([feat_df.reset_index(drop=True), vectorizer_count.reset_index(drop=True), tfidf_word.reset_index(drop=True), 
                             tfidf_ngram.reset_index(drop=True), readbility.reset_index(drop=True), nercount.reset_index(drop=True), 
                             lda_df.reset_index(drop=True)], axis=1)
    print(feature_set.shape)
    feature_set.to_csv('./data/features_{0}.csv'.format('GT' if source==0 else 'ChatGPT'))

    # Grammar check
    trainDF['text_error_length'] = trainDF[selected_col].apply(count_grammar_error)

#feature_extraction(source=0)   #process TEXT column in './data/chatgpt_generated_wiki_data_1_5000.csv'
feature_extraction(source=1)   #process GPT_Generated_Text column in './data/chatgpt_generated_wiki_data_1_5000.csv'