# source:https://medium.com/@adriensieg/text-similarities-da019229c894
import numpy as np
import scipy

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


# def cosine_distance_wordembedding_method(s1, s2):
#     model = loadGloveModel(gloveFile)
#     vector_1 = np.mean([model[word] for word in preprocess(s1)], axis=0)
#     vector_2 = np.mean([model[word] for word in preprocess(s2)], axis=0)
#     cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
#     print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
#           round((1 - cosine) * 100, 2), '%')


def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances
