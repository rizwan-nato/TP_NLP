from __future__ import division
import argparse
# from copyreg import pickle
import pickle
from operator import neg
import pandas as pd
import spacy
import numpy as np
import re
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

__authors__ = ['Rizwan Nato', 'Philippe Formont', 'Pauline Berberi', 'Zineb Lahrichi']
__emails__ = ['rizwan.nato@student-cs.fr', 'philippe.formont@student-cs.fr', 'pauline.berberi@student-cs.fr',
              'zineb.lahrichi@student-cs.fr']


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    nlp = spacy.load("en_core_web_sm")
    with open(path) as f:
        for l in f:
            sentence = []
            doc = nlp(l.lower())
            for token in doc:
                if token.is_alpha:
                    sentence.append(token.lemma_)
            sentences.append( sentence )
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


def sigma(x,y):
    return 1/(1+np.exp(-x.dot(y)))


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5, epochs = 2, lr = 1e-3):
        self.w2id = {}  # word to ID mapping
        self.trainset = sentences  # set of sentences
        self.vocab = {}  # list of valid words and the P(w)
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.nEmbed = nEmbed
        self.loss = []
        self.accLoss = 0
        self.trainWords = 0
        self.minCounts = minCount
        self.epochs = epochs
        self.lr = lr

        count = {}
        # Count all the occurence of the words
        for sentence in self.trainset:
            for word in sentence:
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1

        for word in count:
            if count[word] >= self.minCounts:
                self.vocab[word] = count[word]
        # Create a mapping from word to id and compute P(w)
        for id, word in enumerate(self.vocab):
            self.w2id[word] = id
            self.vocab[word] = self.vocab[word] ** (3 / 4)

        total = sum(self.vocab.values())
        for word in self.vocab:
            self.vocab[word] = self.vocab[word] / total

        #Initialize the embeddings randomly
        self.n_vocab = len(self.vocab.values())
        self.W = np.random.random(size=(self.n_vocab, self.nEmbed)) * 0.1 # Start with a better loss 
        self.C = np.random.random(size=(self.n_vocab, self.nEmbed)) * 0.1


    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        list_to_choose = list(self.vocab.keys())
        probability = list(self.vocab.values())
        neg_ids = []
        while len(neg_ids) < self.negativeRate:
            word = np.random.choice(list_to_choose, p=probability)
            id = self.w2id[word]
            if id not in omit:
                neg_ids.append(id)
        return neg_ids

    def train(self):
        for i in range(self.epochs):
            print(f"Training Epoch {i+1}")
            for counter, sentence in enumerate(self.trainset):
                sentence = list(filter(lambda word: word in self.vocab, sentence))
                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds)
                        self.trainWords += 1
                if counter % 100 == 0:
                    print(' > training %d of %d' % (counter, len(self.trainset)))

                    self.loss.append(self.accLoss / self.trainWords)
                    print(f'    > loss {self.loss[-1]}')
                    self.trainWords = 0
                    self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        vc = self.C[contextId]
        vw = self.W[wordId]
        # According to the paper, function to maximize: $log \sigma(v_c . v_w) + \sum_{negative_context} log \sigma (-v_c_neg . v.w)$
        # We can define the following loss:
        word_context = sigma(vw, vc)
        loss = -(np.log(word_context))

        # Backpropagate the gradient. The training parameters are vw, vc and all the vc for negative sampling
        gradient_vw = (word_context-1) * vc
        gradient_vc = (word_context-1) * vw
        for neg_id in negativeIds:
            vc_neg = self.C[neg_id]
            word_neg_context = sigma(-vw, vc_neg)
            loss -= np.log(word_neg_context)
            gradient_vw += (1 - word_neg_context) * vc_neg
            gradient_c_neg = (1 - word_neg_context) * vw
            self.C[neg_id] -= self.lr * gradient_c_neg
        
        self.C[contextId] -= self.lr * gradient_vc
        self.W[wordId] -= self.lr * gradient_vw
        
        self.accLoss += loss


    def save(self, path):
        data = {
            'trainset': self.trainset,
            'w2id': self.w2id,
            'W': self.W,
            'C': self.C,
            'vocab': self.vocab,
            'negativeRate': self.negativeRate,
            'winSize': self.winSize,
            'nEmbed': self.nEmbed,
            'loss': self.loss,
            'minCounts': self.minCounts,
            'epochs': self.epochs,
            'lr': self.lr
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def similarity(self, word1, word2):
        """
        Computes similarity between the two words. Unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if word1 in self.vocab and word2 in self.vocab:
            ind1 = self.w2id[word1]
            ind2 = self.w2id[word2]

            cosine_sim = np.sum(self.W[ind1]*self.W[ind2]) / (
                        np.linalg.norm(self.W[ind1]) * np.linalg.norm(self.W[ind2]))
            return cosine_sim
        else:
            return 0.5

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sg = SkipGram(sentences=data["trainset"])
        sg.W = data["W"]
        sg.vocab = data["vocab"]
        sg.w2id = data["w2id"]
        return sg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences, minCount=1, nEmbed=300, epochs=1, lr=1e-2)
        sg.train()
        sg.save(opts.model)

    else:
        compteur = 0
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        Y_true = []
        Y_pred = []
        for a, b, y_true in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            pred = sg.similarity(a,b)
            if pred == 0.5:
                compteur += 1
            print(pred, y_true)
            Y_pred.append(pred*10)
            Y_true.append(y_true)
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)
        print("Errors")
        print(mean_squared_error(Y_true, Y_pred))
        print(mean_squared_error(Y_true, [5 for y in Y_pred]))
        print(compteur)
