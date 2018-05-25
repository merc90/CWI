from sklearn.svm           import SVC
import nltk
from nltk.corpus           import wordnet as wn
from utils.dataset         import Dataset
from collections           import Counter
import pyphen
import spacy


class CWI(object):
    def __init__(self, language):

        self.language = language

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if  self.language == 'english':
            self.avg_word_length = 5.3
            self.syl = pyphen.Pyphen(lang='en')
            self.nlp = spacy.load('en_core_web_lg')

        elif self.language == 'spanish':
            self.avg_word_length = 6.2
            self.syl = pyphen.Pyphen(lang='es')
            self.nlp = spacy.load('es_core_news_md')
 
        self.model = SVC()

    def extractFeatures(self, word):
        feature = [] 
        feature.append(len(word)/self.avg_word_length)
        feature.append(len(word.split()))
        feature += list(self.nlp(word).vector)
        feature.append(len(self.syl.inserted(word).split('-')))
        if  self.language == 'english':
            feature.append(len(wn.lemmas(word)))
            feature.append(len(wn.synsets(word)))
            feature.append(sum(word.count(c) for c in ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']))
            feature.append(sum(word.count(c) for c in ['a','e','i','o','u']))
        
        return feature


    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extractFeatures(sent['target_word'].lower()))
            y.append(sent['gold_label'])
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extractFeatures(sent['target_word'].lower()))
        # print("Word_len")
        # print("Span")
        # print("Tocken_len")
        # print("Span")
        # print("Sylla")
        # print("Span")
        # print("Lemme")
        # print("Syns")
        # print("Emb")
        # print("Span")
        # print("Cons")
        # print("Vowels")
        
        return self.model.predict(X)