# This code uses  the OCTIS topic modeling models
# you need to install the package before running: pip install octis
import os
from os import environ, makedirs
from os.path import exists, expanduser, join, splitext
import pickle
import sys
import codecs
import shutil
import requests
import json
import pandas as pd
import numpy as np
from octis.dataset.dataset import Dataset
from octis.models.model import save_model_output, load_model_output
from octis.models.NeuralLDA import NeuralLDA
from sklearn.feature_extraction.text import CountVectorizer

def load_custom_dataset_from_folder(dataset, path, multilabel=False):
    """
    Loads all the dataset from a folder
    Parameters
    ----------
    path : path of the folder to read
    """
    dataset_path = path
    try:
        if exists(dataset_path + "/metadata.json"):
            dataset._load_metadata(dataset_path + "/metadata.json")
        else:
            dataset.__metadata = dict()
        df = pd.read_csv(dataset_path + "/corpus.tsv", sep='\t', header=None)
        if len(df.keys()) > 1:
            # just make sure docs are sorted in the right way (train - val - test)
            final_df = pd.concat(
                [df[df[1] == 'train'],
                    df[df[1] == 'val'],
                    df[df[1] == 'test']])
            #dataset.__metadata['last-training-doc'] = len(
            #    final_df[final_df[1] == 'train'])
            #dataset.__metadata['last-validation-doc'] = len(
            #    final_df[final_df[1] == 'val']) + len(
            #        final_df[final_df[1] == 'train'])
            dataset.__corpus = [d.split() for d in final_df[0].tolist()]
            if len(final_df.keys()) > 2:
                if multilabel:
                    dataset.__labels = [
                        doc.split() for doc in final_df[2].tolist()]
                else:
                    dataset.__labels = final_df[2].tolist()
        else:
            dataset.__corpus = [d.split() for d in df[0].tolist()]
            #dataset.__metadata['last-training-doc'] = len(df[0])
        
        if exists(dataset_path + "/vocabulary.txt"):
            dataset._load_vocabulary(dataset_path + "/vocabulary.txt")
        else:
            vocab = set()
            for d in dataset.__corpus:
                for w in set(d):
                    vocab.add(w)
            dataset.__vocabulary = list(vocab)
        if exists(dataset_path + "/indexes.txt"):
            dataset._load_document_indexes(dataset_path + "/indexes.txt")
        return dataset
    except:
        raise Exception("error in loading the dataset:" + dataset_path)
        return None

def train_save_neurallda(num_topics=20, dataset_path = './data/wiki_5k_gt', out_folder='./model'):
    #dataset_path = './data/wiki_5k_gt'
    #df = pd.read_csv(dataset_path + "/corpus.tsv", sep='\t', header=None)
    #df = df[df[0].apply(lambda x: len(str(x)) >= 100)]
    #df.to_csv(dataset_path + '/corpus.tsv', sep='\t', header=None, index=None)

    dataset = Dataset()
    #dataset.fetch_dataset('20Newsgroup')
    #download_dataset('20Newsgroup', target_dir='./20Newsgroup', cache_path=None)
    #dataset.load_custom_dataset_from_folder("./20Newsgroup")
    dataset.load_custom_dataset_from_folder(dataset_path) #wikipedia 5000 articles ground truth

    # print statistics of the dataset
    print(len(dataset._Dataset__corpus))
    print(len(dataset._Dataset__vocabulary))
    #print(dataset._Dataset__corpus[0:5])

    #train a Neural LDA on the 20 news group dataset
    model = NeuralLDA(num_topics=num_topics)
    trained_model = model.train_model(dataset)

    #after training, prints list of topics and topic document matrix
    print(trained_model.keys())
    #for topic in trained_model['topics']:
    #    print(" ".join(topic))
    #print(trained_model['topic-document-matrix'])

    #evaluate the metrics
    from octis.evaluation_metrics.coherence_metrics import Coherence
    cv = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_v')
    print('Coherence: ' + str(cv.score(trained_model)))

    from octis.evaluation_metrics.diversity_metrics import TopicDiversity
    diversity = TopicDiversity(topk=10)
    print('Diversity score: '+str(diversity.score(trained_model)))

    #save trained model outputs
    os.makedirs(out_folder, exist_ok=True)
    save_model_output(trained_model, path = out_folder+'/model')
    id2vocab = {i: w for i, w in enumerate(dataset.get_vocabulary())}
    json_vocab = json.dumps(id2vocab)
    with open(out_folder+'/vocab2id.json', 'w', encoding='utf8') as outfile:
        outfile.write(json_vocab)

    #get topic document matrix for train-validation-test
    tdm = np.asarray(trained_model['topic-document-matrix'])
    print(tdm.shape)
    #get validation test result
    train, validation, test = dataset.get_partitioned_corpus(use_validation=True)
    data_corpus_train = [' '.join(i) for i in train]
    data_corpus_test = [' '.join(i) for i in test]
    data_corpus_validation = [' '.join(i) for i in validation]
    vocab = dataset.get_vocabulary()
    x_train, x_test, x_valid, input_size = model.preprocess(vocab, data_corpus_train, test=data_corpus_test, 
                                                                validation=data_corpus_validation)
    val_results = model.inference(x_valid)
    vtdm = np.asarray(val_results['test-topic-document-matrix'])
    print(vtdm.shape)
    ttdm = np.asarray(trained_model['test-topic-document-matrix'])
    print(ttdm.shape)
    final_tdm = np.concatenate([tdm, vtdm, ttdm], axis=1).transpose()
    print(final_tdm.shape)
    feature_cols = ["lda{:02d}".format(i) for i in range(num_topics)]
    lda_df = pd.DataFrame(final_tdm, columns=feature_cols)
    #return topic modeling features
    return model, trained_model, lda_df

if __name__ == "__main__":
    model, trained_model, lda_df = train_save_neurallda(dataset_path = './data/wiki_5k_gt', out_folder='./model/lda_wiki_5k_gt')
    #trained_model = load_model_output(output_path='./model/lda_wiki_5k_gt/model.npz', vocabulary_path='./model/lda_wiki_5k_gt/vocab2id.json', top_words=20)
    #model, trained_model, lda_df = train_save_neurallda(dataset_path = './data/wiki_5k_chatgpt', out_folder='./model/lda_wiki_5k_chatgpt')
    #print loaded model components
    print(trained_model.keys())
    for topic in trained_model['topics']:
        print(" ".join(topic))
        #print(" ".join([t[0] for t in topic]))
    #print lda features
    print(lda_df.shape)