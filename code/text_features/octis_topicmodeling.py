# This code uses  the OCTIS topic modeling models
# you need to install the package before running: pip install octis
from os import environ, makedirs
from os.path import exists, expanduser, join, splitext
import pickle
import sys
import codecs
import shutil
import requests
import json
from octis.dataset.dataset import Dataset

dataset = Dataset()
#dataset.fetch_dataset('20Newsgroup')
#download_dataset('20Newsgroup', target_dir='./20Newsgroup', cache_path=None)
#dataset.load_custom_dataset_from_folder("./20Newsgroup")
dataset.load_custom_dataset_from_folder("./data/wiki_5k_gt") #wikipedia 5000 articles ground truth

# print statistics of the dataset
print(len(dataset._Dataset__corpus))
print(len(dataset._Dataset__vocabulary))
print(dataset._Dataset__corpus[0:5])

#train a Neural LDA on the 20 news group dataset
from octis.models.NeuralLDA import NeuralLDA
model = NeuralLDA(num_topics=20)
trained_model = model.train_model(dataset)

#after training, prints 
print(trained_model.keys())
for topic in trained_model['topics']:
    print(" ".join(topic))

#evaluate the metrics
from octis.evaluation_metrics.coherence_metrics import Coherence
cv = Coherence(texts=dataset.get_corpus(),topk=10, measure='c_v')
print('Coherence: ' + str(cv.score(trained_model)))

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
diversity = TopicDiversity(topk=10)
print('Diversity score: '+str(diversity.score(trained_model)))

