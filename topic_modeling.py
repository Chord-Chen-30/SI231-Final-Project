# Code refered to https://github.com/mikemikezhu/topic_modelling_covid19
# Date 2022-6-12

from hashlib import new
from scipy import rand
import torch
import csv
import re
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy


def get_LSA_embedding(words_to_show_on_2D):

    corpus = []

    with open('./metadata.csv') as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)

        for lines in csv_reader:
            abstract = lines[8]
            corpus.append(abstract)

    print('Total corpus extracted: {}'.format(len(corpus)))
    print('Example:')
    for i in range(2):
        sample = corpus[i]
        print('{}: {}'.format((i + 1), sample))


    corpus = [x.lower() for x in corpus]

    corpus = [re.sub(r'[^a-z0-9 ]+', '', x) for x in corpus]

    print('After data clean-up:')
    for i in range(5):
        sample = corpus[i]
        print('{}: {}'.format((i + 1), sample))


    stop_words = list(stopwords.words('english'))
    stop_words += list(stopwords.words('french'))
    stop_words += list(stopwords.words('spanish'))

    print(len(stop_words))


    vectorizer = TfidfVectorizer(stop_words=stop_words, 
                                max_features=1000)
    term_document_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names()

    vocabulary_LSA = vectorizer.vocabulary_

    key_list = list(vocabulary_LSA.keys())
    key_list.sort()
    print(key_list)


    # SVD represent documents and terms in vectors
    svd_model = TruncatedSVD(n_components=10, random_state=0)
    embedding_LSA = svd_model.fit_transform(term_document_matrix)

    # Check the shape
    print(embedding_LSA.shape)
    print(type(embedding_LSA))

    ret = []
    for w in words_to_show_on_2D:
        idx = vocabulary_LSA[w]
        ret.append(torch.tensor(embedding_LSA[idx]).unsqueeze(0))

    return torch.cat(ret)


def get_BERT_embedding():
    context_1 = 'drug and drugs'
    context_2 = 'best and better'
    context_3 = 'play and plays'
    context_4 = 'produce and products'

    context = [context_1, context_2, context_3, context_4]

    tokenenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    ret = []

    for c in context:
        print(c)
        token = tokenenizer(c, return_tensors="pt")
        embedding = bert(**token).last_hidden_state

        assert embedding.size(1) == 5

        ret.append(embedding[0, 1].unsqueeze(0))
        ret.append(embedding[0, 3].unsqueeze(0))

    return torch.cat(ret).detach()



def pca(matrix, dim=2):

    pca = PCA(n_components=2, random_state=0)
    new_matrix = pca.fit_transform(matrix)

    # tsne = TSNE(n_components=2)
    # new_matrix = tsne.fit_transform(matrix)

    return numpy.array(new_matrix)




def main():

    words_to_show_on_2D = ['drug', 'drugs', 
                           'best', 'better', 
                           'play', 'plays',
                           'production', 'products']

    embedding_LSA = get_LSA_embedding(words_to_show_on_2D)
    embedding_BERT = get_BERT_embedding()

    print(embedding_BERT.shape)    

    pca_LSA = pca(embedding_LSA)
    print(pca_LSA)

    pca_BERT = pca(embedding_BERT)
    print(pca_BERT)


    import matplotlib.pyplot as plt

    plt.figure()
    plt.xlim(-0.3, 0.5)
    plt.ylim(-0.4, 0.4)
    plt.scatter(pca_LSA[:, 0], pca_LSA[:, 1])
    for label, x, y in zip(words_to_show_on_2D, pca_LSA[:, 0], pca_LSA[:, 1]):
        plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0,0),textcoords='offset points')
    plt.savefig('./LSA-2d-visualization')
    plt.close()

    plt.figure()
    plt.scatter(pca_BERT[:, 0], pca_BERT[:, 1])
    for label, x, y in zip(words_to_show_on_2D, pca_BERT[:, 0], pca_BERT[:, 1]):
        plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0,0),textcoords='offset points')
    plt.savefig('./BERT-2d-visualization')
    plt.close()



if __name__ == '__main__':
    main()

    print('done')