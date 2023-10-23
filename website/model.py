import spacy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np
import gensim.downloader as api
from gensim.models.word2vec import LineSentence, Word2Vec # TODO: train on the same model as the corpus used

class Tagger:
    def __init__(self, text, stem):
        self.text = text.lower()
        self.sentences = sent_tokenize(self.text)

        self.ps = PorterStemmer()

        self.nlp = spacy.load('en_core_web_sm')
        # nltk.download('stopwords')
        self.docs = [self.nlp(sentence) for sentence in self.sentences]

        ## Full vocabulary with fequences
        self.vocab = self.__get_vocab(stem=stem)

    def __get_vocab(self, stem=True):
        words = [(token.text, token.pos_) for doc in self.docs for token in doc if token.pos_ != 'SPACE' and token.pos_ != 'PUNCT']

        if stem:
            words = [(self.ps.stem(w), t) for w, t in words]

        return Counter(words)

    def __get_sent_part(self, tag):
        return [(token.text, token.pos_) for doc in self.docs for token in doc if tag in token.dep_]
    
    def __get_extended_sent_part(self, tag):
        def get_subtree_list(token, doc):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1

            return str(doc[start:end]).split()

        return [get_subtree_list(token, doc) for doc in self.docs for token in doc if tag in token.dep_]


    def get_subjects(self):
        return self.__get_sent_part("subj")
    
    def get_extended_subjects(self):
        return self.__get_extended_sent_part("subj")

    def get_objects(self):
        return self.__get_sent_part("obj")
    
    def get_extended_objects(self):
        return self.__get_extended_sent_part("obj")

    def get_verbs(self):
        return self.__get_sent_part("ROOT")
    
    def pred_pos(self, w):
        '''
        Returns a tuple (str, str) of the word and the predicted POS tag
        '''
        return self.nlp(w)[0].pos_
    
    def stem_word(self, w):
        return self.ps.stem(w)

class Predicter:
    def __init__(self, text, stem=False):
        self.text = text
        self.stem = stem
        self.tagger = Tagger(self.text, stem=self.stem)

        # TODO: Or use the extended version instead?
        self.S = Counter(self.tagger.get_subjects())
        self.V = Counter(self.tagger.get_verbs())
        self.O = Counter(self.tagger.get_objects())

        self.SVOs = [self.S, self.V, self.O]
        self.tags = ['subj', 'verb', 'obj']

    
    def __normalize(self, probs):
            probs = np.asarray(list(probs))

            ## In case of all zero values
            if not np.any(probs):
                return probs

            factor = 1 / probs.sum()
            
            return factor * probs

    def __pred_svo(self, feature):
        word, pos = feature

        if pos is None:
            ## Predict POS tag for word if it wasn't given
            pos = self.tagger.pred_pos(word)

        def prob(svo):
            # TODO: Smoothing -> is it good for us? Handle <UNK> vs 'cat is a verb'
            smoothing = False

            c_total = self.tagger.vocab[(word, pos)]
            c_svo = svo[(word, pos)]

            if smoothing:
                v = sum(self.tagger.vocab.values())

                return (c_svo + 1) / (c_total + v)
            else:
                if c_total:
                    return c_svo / c_total
                else:
                    return 0

        probabilies = self.__normalize([prob(svo) for svo in self.SVOs])
        print(probabilies)

        if not np.any(probabilies):
            if pos == "VERB":
                return "verb"
            else:
                probabilies = np.asarray([0.5, 0, 0.5])

        return np.random.choice(self.tags, size=1, p=probabilies)[0]
    
    def __pred_word(self, svo):
        print(svo)

        p = self.__normalize(svo.values())
        words = [w for w,p in svo.keys()]

        return np.random.choice(words, size=1, p=p)[0]

    def pred_sent(self, feature):
        word, pos = feature
        word = word.lower()

        if self.stem:
            word = self.tagger.stem_word(word)

        given_tag = self.__pred_svo((word, pos))

        # TODO: Use embeddings or something to predict the remaining words.
        # TODO: Add determiners at least.
        if given_tag == 'subj':
            S = word
            V = self.__pred_word(self.V)
            O = self.__pred_word(self.O)
        elif given_tag == 'verb':
            S = self.__pred_word(self.S)
            V = word
            O = self.__pred_word(self.O)
        elif given_tag == 'obj':
            S = self.__pred_word(self.S)
            V = self.__pred_word(self.V)
            O = word

        sent = " ".join((S, V, O))
        return sent.capitalize() + "."