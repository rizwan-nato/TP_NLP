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
from tqdm import tqdm

__authors__ = ['Rizwan Nato', 'Philippe Formont', 'Pauline Berberi', 'Zineb Lahrichi']
__emails__ = ['rizwan.nato@student-cs.fr', 'philippe.formont@student-cs.fr', 'pauline.berberi@student-cs.fr',
              'zineb.lahrichi@student-cs.fr']


def text2sentences(path, number_of_line=np.inf):
    # feel free to make a better tokenization/pre-processing
    '''
    Input
      Path: Path of the text file
      Number_of_line: Number of line we want to load. If not specified we will load the complete text file
    '''
    sentences = []
    sentences_lemma = []
    nlp = spacy.load("en_core_web_sm")
    with open(path) as f:
        for counter, l in enumerate(f):
            if counter >= number_of_line:
                break
            sentence = []
            sentence_lemma = []
            doc = nlp(l.lower())
            for token in doc:
                if token.is_alpha: 
                    sentence_lemma.append(token.lemma_)
                    sentence.append(token.text)
            sentences.append( sentence )
            sentences_lemma.append( sentence_lemma )
    return sentences, sentences_lemma


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


def sigma(x,y):
    return 1/(1+np.exp(-x.dot(y)))


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
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
        self.epochs_trained = 0

        count = {}
        # Count all the occurence of the words
        for sentence in self.trainset:
            for word in sentence:
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1
        #Create Unknown token
        self.vocab["<UKN>"] = 0
        for word in count:
            if count[word] >= self.minCounts:
                self.vocab[word] = count[word]
            else:
                self.vocab["<UKN>"] +=1
        # Create a mapping from word to id and compute P(w)
        for id, word in enumerate(self.vocab):
            self.w2id[word] = id
            self.vocab[word] = self.vocab[word] ** (3 / 4)

        total = sum(self.vocab.values())
        for word in self.vocab:
            self.vocab[word] = self.vocab[word] / total

        #Initialize the embeddings randomly. Should not be too big at the start of the training, the loss is too high at the start otherwise
        self.n_vocab = len(self.vocab.values())
        self.W = np.random.random(size=(self.n_vocab, self.nEmbed)) * 0.1 
        self.C = np.random.random(size=(self.n_vocab, self.nEmbed)) * 0.1

        #Create the unigram table for the negative sampling.
        list_to_choose = list(self.vocab.keys())
        list_to_choose = [self.w2id[word] for word in list_to_choose]
        probability = list(self.vocab.values())
        self.unigram_size = int(1e8)
        self.unigram_table = np.random.choice(list_to_choose, size=self.unigram_size, p=probability)

        

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        neg_ids = []
        while len(neg_ids) < self.negativeRate:
            id = self.unigram_table[np.random.randint(0, self.unigram_size)] #Select randomly from unigram table
            if id not in omit:
                neg_ids.append(id)
        return neg_ids

    def train(self, epochs=1, lr=1e-2):
        for i in range(epochs):
            print(f"Training Epoch {self.epochs_trained + 1}")
            for sentence in self.trainset:
                for wpos, word in enumerate(sentence):
                    # If the word or context is not in vocabulary, we remplace it with the unknown token
                    if word in self.vocab:
                        wIdx = self.w2id[word]
                    else:
                        wIdx = self.w2id["<UKN>"]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    for context_word in sentence[start:end]:
                        if context_word in self.vocab:
                            ctxtId = self.w2id[context_word]
                        else:
                            ctxtId = self.w2id["<UKN>"]
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds, lr)
                        self.trainWords += 1
                

            self.loss.append(self.accLoss / self.trainWords)
            print(f'    > loss {self.loss[-1]}')
            self.epochs_trained +=1
            self.trainWords = 0
            self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds, lr):
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
            self.C[neg_id] -= lr * gradient_c_neg
        
        self.C[contextId] -= lr * gradient_vc
        self.W[wordId] -= lr * gradient_vw
        
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
            'epochs_trained': self.epochs_trained,
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
        # If the words are not in the vocabulary, we take the embedding of the unkonwn token. Then compute the cosine similarity.
        unknown = False
        if word1 in self.vocab:
            w1_emb = self.W[self.w2id[word1]]
        else:
            w1_emb = self.W[self.w2id["<UKN>"]]
            unknown = True
        
        if word2 in self.vocab:
            w2_emb = self.W[self.w2id[word2]]
        else:
            w2_emb = self.W[self.w2id["<UKN>"]]
            unknown = True
        cosine = np.sum(w1_emb*w2_emb) / (np.linalg.norm(w1_emb)*np.linalg.norm(w2_emb))
        return cosine, unknown
        

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sg = SkipGram(sentences=data["trainset"], nEmbed=data["nEmbed"], negativeRate=data["negativeRate"], winSize=data["winSize"], minCount=data["minCounts"])
        sg.W = data["W"]
        sg.C = data["C"]
        sg.loss = data["loss"]
        sg.epochs_trained = data["epochs_trained"]
        return sg

def run_training(sg, Number_of_epochs, Learning_rate_Schedule, early_stopping = 1e-3, name="model"):
    for i, lr in zip(range(Number_of_epochs), Learning_rate_Schedule):
        sg.train(epochs=1, lr=lr)
        sg.save(name)
        pairs = loadPairs("simlex.csv")
        if i > 0:
            if sg.loss[-2] - sg.loss[-1] < early_stopping:
                print("Early stopping..")
                break;

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        Learning_rate_Schedule =  [1e-2]*200
        Number_of_epochs = 200
        nEmbed = 300
        winSize = 9
        minCount = 2
        sentences_no_lemma, sentences_with_lemma = text2sentences(opts.text)
        sg = SkipGram(sentences=sentences_with_lemma, minCount=minCount, nEmbed=nEmbed, winSize=winSize, negativeRate=5)
        run_training(sg, Number_of_epochs, Learning_rate_Schedule, early_stopping=5*1e-3, name=opts.model)

    else:
        compteur = 0
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        Y_true = []
        Y_pred = []
        for a, b, y_true in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            pred = sg.similarity(a,b)
            print(pred[0])

