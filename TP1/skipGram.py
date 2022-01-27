from __future__ import division
import argparse
import pandas as pd
import spacy
import numpy as np

__authors__ = ['Rizwan Nato', 'Philippe Formont', 'Pauline Berberi', 'Zineb Lahrichi']
__emails__ = ['rizwan.nato@student-cs.fr', 'philippe.formont@student-cs.fr', 'pauline.berberi@student-cs.fr',
              'zineb.lahrichi@student-cs.fr']


def text2sentences(path):
    sp = spacy.load('en_core_web_sm')
    sentences = []
    with open(path) as f:
        for l in f:
            sentence = sp(l)
            sentences.append(sentence)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.w2id = {}  # word to ID mapping
        self.trainset = sentences  # set of sentences
        self.vocab = {}  # list of valid words and the P(w)
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.nEmbed = nEmbed
        self.loss = None
        self.minCounts = minCount

        self.W = np.random.random(size=(self.nEmbed, len(self.vocab)))
        self.C = np.random.random(size=(self.nEmbed, len(self.vocab)))

        # Count all the occurence of the words
        for sentence in self.trainset:
            for word in sentence:
                word = word.text.lower()
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1

        # Create a mapping from word to id and compute P(w)
        for id, word in enumerate(self.vocab):
            self.w2id[word] = id
            self.vocab[word] = self.vocab[word] ** (3 / 4)

        total = sum(self.vocab.values())
        for word in self.vocab:
            self.vocab[word] = self.vocab[word] / total

        self.n_vocab = len(self.vocab.values())

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        list_to_choose = list(self.vocab.keys())
        probability = list(self.vocab.values())
        for index in omit:
            probability[index] = 0
        probability = np.array(probability)
        probability = probability / np.sum(probability)
        return np.random.choice(list_to_choose, self.negativeRate, p=probability)

    def train(self):
        self.loss = []
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)

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

            if counter % 1000 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        raise NotImplementedError('here is all the fun!')

    def save(self, path):
        raise NotImplementedError('implement it!')

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

            cosine_sim = (self.W[ind1].dots(self.W[ind2])) / (
                        np.linalg.norm(self.W[ind1]) * np.linalg.norm(self.W[ind2]))
            return cosine_sim
        else:
            return 0

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
