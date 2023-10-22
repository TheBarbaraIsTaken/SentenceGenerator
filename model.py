# Import libraries
import spacy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np

# Defining a class Tagger with functions to POS tag the corpus
class Tagger:
    # pre processing the corpus (put letters into lower case and get tokens)
    def __init__(self, text):
        self.text = text.lower()
        self.sentences = sent_tokenize(self.text)
        # remove stopwords
        self.nlp = spacy.load('en_core_web_sm')
        # nltk.download('stopwords')
        self.docs = [self.nlp(sentence) for sentence in self.sentences]

        # Full vocabulary with frequences
        self.vocab = self.__get_vocab(stem=False) 
    # create a stemmed vocabulary list and return the number of words in it
    def __get_vocab(self, stem=True):
        words = [(token.text, token.pos_) for doc in self.docs for token in doc if token.pos_ != 'SPACE' and token.pos_ != 'PUNCT']

        if stem:
            ps = PorterStemmer()
            words = [(ps.stem(w), t) for w, t in words]

        return Counter(words)
    # get the POS tag for a token
    def __get_sent_part(self, tag):
        return [(token.text, token.pos_) for doc in self.docs for token in doc if tag in token.dep_]
    
    def __get_extended_sent_part(self, tag):
        def get_subtree_list(token, doc):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1

            return str(doc[start:end]).split()

        return [get_subtree_list(token, doc) for doc in self.docs for token in doc if tag in token.dep_]

    # get the subjects
    def get_subjects(self):
        return self.__get_sent_part("subj")
    
    def get_extended_subjects(self):
        return self.__get_extended_sent_part("subj")
    # get the objects
    def get_objects(self):
        return self.__get_sent_part("obj")
    
    def get_extended_objects(self):
        return self.__get_extended_sent_part("obj")
    # get the verbs
    def get_verbs(self):
        return self.__get_sent_part("ROOT")
    
    def pred_pos(self, w):
        '''
        Returns a tuple (str, str) of the word and the predicted POS tag
        '''
        return self.nlp(w)[0].pos_

# Defining the class Predicter with functions to predict the sentence out of the word and/or the POS tag
class Predicter:
    # get the POS tags for the words in the corpus
    def __init__(self, text):
        self.text = text
        self.tagger = Tagger(self.text)

        # TODO: Or use the extended version instead?
        self.S = Counter(self.tagger.get_subjects())
        self.V = Counter(self.tagger.get_verbs())
        self.O = Counter(self.tagger.get_objects())

        self.SVOs = [self.S, self.V, self.O]
        self.tags = ['subj', 'verb', 'obj']

    
    def __normalize(self, probs):
            probs = np.asarray(list(probs))
            factor = 1 / probs.sum()

            return factor * probs
    
    def __pred_svo(self, feature):
        word, pos = feature

        if pos is None:
            ## Predict POS tag for word if it wasn't given
            pos = self.tagger.pred_pos(word)

        def prob(svo):
            # TODO: Smoothing -> is it good for us? Handle <UNK> vs 'cat is a verb'
            c_total = self.tagger.vocab[(word, pos)]
            c_svo = svo[(word, pos)]

            v = sum(self.tagger.vocab.values())

            if (c_total + v):
                return (c_svo + 1) / (c_total + v)
            else:
                return 0

        probabilies = self.__normalize([prob(svo) for svo in self.SVOs])

        return np.random.choice(self.tags, size=1, p=probabilies)[0]
    
    def __pred_word(self, svo):
        p = self.__normalize(svo.values())
        words = [w for w,p in svo.keys()]

        return np.random.choice(words, size=1, p=p)[0]

    def pred_sent(self, feature):
        word, pos = feature
        given_tag = self.__pred_svo(feature)

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