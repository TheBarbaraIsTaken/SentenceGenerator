import spacy
from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
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
        return [(token.text, token.pos_) for doc in self.docs for token in doc if tag in token.dep_ and token.pos_ != 'SPACE' and token.pos_ != 'PUNCT']
    
    def __get_extended_sent_part(self, tag):
        def get_subtree_list(token, doc):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1

            return str(doc[start:end]).split()

        return [get_subtree_list(token, doc) for doc in self.docs for token in doc if tag in token.dep_]


    def get_subjects(self):
        s = self.__get_sent_part("subj")
        return [(w,p) for w,p in s if p == 'NOUN' or p == 'PROPN' or p == 'PRON']
    
    def get_extended_subjects(self):
        s = self.__get_extended_sent_part("subj")
        return [(" ".join(l), 'NOUN') for l in s if len(l) <= 2]

    def get_objects(self):
        o = self.__get_sent_part("obj")
        return [(w,p) for w,p in o if p == 'NOUN' or p == 'PROPN' or p == 'PRON']
    
    def get_extended_objects(self):
        o = self.__get_extended_sent_part("obj")
        return [(" ".join(l), 'NOUN') for l in o if len(l) <= 2]

    def get_verbs(self):
        v = self.__get_sent_part("ROOT")
        return [(w,p) for w,p in v if p == 'VERB' or p == 'AUX']
    
    def get_extended_verbs(self):
        phrases = []

        for doc in self.docs:
            for token in doc:
                verb_phrase = []
                verb_phrase.append(token.text)


                if token.dep_ == 'ROOT' and (token.pos_ == "VERB" or token.pos_ == "AUX"): 
                    for compverb in list(token.lefts)[::-1]:
                        if compverb.pos_ not in ("AUX", "VERB"):
                            break
                        
                        verb_phrase.append(compverb.text)

                    phrases.append((" ".join(verb_phrase[::-1]), "VERB"))

        return phrases
    
    def pred_pos(self, w):
        '''
        Returns a tuple (str, str) of the word and the predicted POS tag
        '''
        return self.nlp(w)[0].pos_
    
    def stem_word(self, w):
        return self.ps.stem(w)

class Predicter:
    def __init__(self, f_path, stem=False):
        with open(f_path, 'r') as f:
            self.text = f.read()

        self.stem = stem
        self.tagger = Tagger(self.text, stem=self.stem)

        # TODO: Or use the extended version instead?
        self.S = Counter(self.tagger.get_subjects())
        self.V = Counter(self.tagger.get_verbs())
        self.O = Counter(self.tagger.get_objects())

        self.SVOs = [self.S, self.V, self.O]
        self.tags = ['subj', 'verb', 'obj']


        ## Train embeddings
        # self.w2v = api.load("word2vec-google-news-300")
        
        vector_size = 100
        window = 5
        min_count = 5
        sg = 0
        negative = 30

        #line_sentences=LineSentence(f_path)

        #self.w2v = Word2Vec(sentences=line_sentences, vector_size=vector_size, alpha=0.025, window=window, min_count=min_count, max_vocab_size=None, 
        #             sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=sg, hs=0, negative=negative, ns_exponent=0.75,
        #             cbow_mean=1, epochs=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, 
        #             compute_loss=False, callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True) 

        self.w2v = Word2Vec.load("word2vec.model")


        print("Initialization is ready")

    
    def __normalize(self, probs):
            probs = np.asarray(list(probs))
            probs = np.absolute(probs)

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

        if pos == "VERB":
            return "verb"

        def prob(svo):
            # Smoothing -> is it good for us? Handle <UNK> vs 'cat is a verb'
            smoothing = False

            c_total = self.tagger.vocab[(word, pos)]
            c_svo = svo[(word, pos)] # TODO: extend so svo can contain adjectives etc.

            if smoothing:
                v = sum(self.tagger.vocab.values())

                return (c_svo + 1) / (c_total + v)
            else:
                if c_total:
                    return c_svo / c_total
                else:
                    return 0

        probabilies = self.__normalize([prob(svo) for svo in self.SVOs])

        if not np.any(probabilies):
            probabilies = np.asarray([0.5, 0, 0.5])

        return np.random.choice(self.tags, size=1, p=probabilies)[0]
    
    def __pred_word_embedding(self, svo, *target):
        def cosine_similarity(w, target):
            try:
                vec1 = self.w2v.wv[w]
                vec2 = self.w2v.wv[target]
                return vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            except KeyError:
                return 1
            
        if len(target) == 1:
            # p = self.__normalize(svo.values())
            p = self.__normalize([(1-abs(cosine_similarity(w, target[0])))*svo[(w,p)] for w,p in svo.keys()])
        else:
            p = self.__normalize([(1-abs(cosine_similarity(w, target[0])))*(1-abs(cosine_similarity(w, target[1])))*svo[(w,p)] for w,p in svo.keys()])

        words = [w for w,p in svo.keys()]

        if not np.any(p):
            return np.random.choice(words, size=1)[0]
        
        return np.random.choice(words, size=1, p=p)[0]

    def pred_sent(self, feature):
        word, pos = feature
        word = word.lower()

        if self.stem:
            word = self.tagger.stem_word(word)

        given_tag = self.__pred_svo((word, pos))

        # Use embeddings or something to predict the remaining words.
        # TODO: Add determiners at least.
        if given_tag == 'subj':
            S = word
            V = self.__pred_word_embedding(self.V, word)
            O = self.__pred_word_embedding(self.O, word, V)
        elif given_tag == 'verb':
            S = self.__pred_word_embedding(self.S, word)
            V = word
            O = self.__pred_word_embedding(self.O, word, V)
        elif given_tag == 'obj':
            S = self.__pred_word_embedding(self.S, word)
            V = self.__pred_word_embedding(self.V, word, S)
            O = word

        sent = " ".join((S, V, O))
        return sent.capitalize() + "."