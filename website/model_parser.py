from nltk.tokenize import sent_tokenize
import spacy
import numpy as np
from collections import Counter
import gensim.downloader as api # For embeddings
from . import dependency_parser as dp

class Predicter:
    def __init__(self, f_path):
        ## Constants
        self.VERB_TAGS = {"VERB", "AUX"}
        self.SKIP_TAGS = {"PUNCT", "SPACE", "SYM", "X"}
        self.SUBJ_TAGS = {'NOUN', 'PROPN', 'PRON'}

        self.nlp = spacy.load("en_core_web_sm")
        self.w2v = api.load("word2vec-google-news-300")

        ## Precess corpus data
        self.f_path = f_path
        self.docs = dp.read_file(self.f_path)

        ## Compute vocabularies
        self.S, self.O, self.vocab = self.__get_vocabs()

        ## Compute co-occurence matrices
        self.sv_subject_dict, self.sv_verb_dict, self.sv_matrix = self.__get_sv_matrix()
        self.vo_verb_dict, self.vo_object_dict, self.vo_matrix = self.__get_vo_matrix()


        print("Initialization is ready")

    def __get_sv_matrix(self):
        ## Initialize empty dictionaries for subjects and verbs
        sv_subject_dict = {}
        sv_verb_dict = {}

        ## Loop over docs and build the (subject, verb) pairs and dictionaries
        pairs = []
        for doc in self.docs:
            subject_verb_pairs = dp.extract_subject_verb_pairs(doc)
            pairs.extend(subject_verb_pairs)

            for subject, verb in subject_verb_pairs:
                if subject not in sv_subject_dict:
                    sv_subject_dict[subject] = len(sv_subject_dict)
                if verb not in sv_verb_dict:
                    sv_verb_dict[verb] = len(sv_verb_dict)

        ## Initialize the numpy matrix with zeros
        num_subjects = len(sv_subject_dict)
        num_verbs = len(sv_verb_dict)
        sv_matrix = np.zeros((num_subjects, num_verbs), dtype=int)

        ## Populate the matrix with counts of (subject, verb) pairs
        for subject, verb in pairs:
            subject_index = sv_subject_dict[subject]
            verb_index = sv_verb_dict[verb]
            sv_matrix[subject_index, verb_index] += 1

        return sv_subject_dict, sv_verb_dict, sv_matrix

    def __get_vo_matrix(self):
        # Initialize empty dictionaries for objects and verbs
        vo_verb_dict = {}
        vo_object_dict = {}

        # Loop over docs and build the (verb, object) pairs and dictionaries
        pairs = []
        for doc in self.docs:
            verb_object_pairs = dp.extract_verb_object_pairs(doc)
            pairs.extend(verb_object_pairs)

            for verb, object in verb_object_pairs:
                if verb not in vo_verb_dict:
                    vo_verb_dict[verb] = len(vo_verb_dict)
                if object not in vo_object_dict:
                    vo_object_dict[object] = len(vo_object_dict)

        # Initialize the numpy matrix with zeros
        num_verbs = len(vo_verb_dict)
        num_objects = len(vo_object_dict)
        vo_matrix = np.zeros((num_verbs, num_objects), dtype=int)

        # Populate the matrix with counts of (verb, object) pairs
        for verb, object in pairs:
            verb_index = vo_verb_dict[verb]
            object_index = vo_object_dict[object]
            vo_matrix[verb_index, object_index] += 1
        
        return vo_verb_dict, vo_object_dict, vo_matrix

    def __get_vocabs(self):
        S = []
        O = []
        vocab = []

        for doc in self.docs:
            for token in doc:
                if token["upostag"] not in self.SKIP_TAGS:
                    ## Add to vocabulary 
                    vocab.append((token["form"].lower(), token["upostag"]))

                    ## Add to subjects
                    if "subj" in token["deprel"]:
                        subject = dp.get_subject_phrase(token, doc)

                        for subject_token in subject:
                            S.append((subject_token["form"].lower(), subject_token["upostag"]))
                    
                    ## Add to objects
                    if "obj" in token["deprel"]:
                        object = dp.get_object_phrase(token, doc)

                        for object_token in object:
                            O.append((object_token["form"].lower(), object_token["upostag"]))
        
        return Counter(S), Counter(O), Counter(vocab)
    
    def __normalize(self, probs):
            probs = np.asarray(list(probs))

            ## In case of all zero values
            if not np.any(probs):
                return probs
            
            factor = 1 / probs.sum()
            
            return factor * probs
    
    def pred_pos(self, word):
        return self.nlp(word)[0].pos_
    
    def pred_svo(self, feature):
        word, pos = feature
        
        ## Predict POS tag for word if it wasn't given
        if pos is None:
            pos = self.pred_pos(word)

        if pos in self.VERB_TAGS:
            return "verb"
        
        def prob(svo):
            # TODO: use smoothing between subject and object?
            c_total = self.vocab[(word, pos)]
            c_svo = svo[(word, pos)]

            if c_total:
                return c_svo / c_total
            else:
                return 0
        
        probabilies = self.__normalize([prob(svo) for svo in (self.S, self.O)])

        ## In case of all zero probabilities
        if not np.any(probabilies):
            return np.random.choice(("subj", "obj"), size=1)[0]  # uniform

        return np.random.choice(("subj", "obj"), size=1, p=probabilies)[0]
    
    def cosine_similarity(self, w1, w2):
        try:
            w1_vec = self.w2v[w1]
            w2_vec = self.w2v[w2]

            cos = w1_vec.dot(w2_vec)/(np.linalg.norm(w1_vec)*np.linalg.norm(w2_vec))

            ## TODO: Deal with negative similarity
            if cos < 0:
                return 1
            
            return cos
        except KeyError:
            return 1
    
    def __pred_sv(self, S):
        ## Extract the root of subject
        # TODO: test this
        try:
            s = [token.text for token in self.nlp(S) if token.dep_ == "ROOT"][0]
        except IndexError:
            s = str(S)

        def pred_v(s_index):
            ## Use the maximum value
            # v_index = np.argmax(self.sv_matrix[s_index, :])
            # return list(self.sv_verb_dict.keys())[v_index]

            ## Use probability
            probabilies = self.__normalize(self.sv_matrix[s_index, :])

            ## All zero probabilities
            if not np.any(probabilies):
                return np.random.choice(list(self.sv_verb_dict.keys()), size=1)[0]  # uniform

            return np.random.choice(list(self.sv_verb_dict.keys()), size=1, p=probabilies)[0]

        # TODO: Make it more random with using probability for embeddings?
        if s in self.sv_subject_dict:
            print(s, "subject in vocabulary")

            s_index = self.sv_subject_dict[s]
            return pred_v(s_index)
        else:
            if s in self.w2v:
                print(s, "subject in embeddings")

                ## Get the most similar subject from vocabulary
                s_in_vocab = min(self.sv_subject_dict.keys(), key=lambda w: self.cosine_similarity(s, w))
                s_index = self.sv_subject_dict[s_in_vocab]

                return pred_v(s_index)
            else:
                print(s, "subject in unkown")
                return np.random.choice(list(self.sv_verb_dict.keys()), size=1)[0]

    def __pred_vo(self, V):
        def pred_o(v_index):
            ## Use the maximum value
            #o_index = np.argmax(self.vo_matrix[v_index, :])
            #return list(self.vo_object_dict.keys())[o_index]

            ## Use probability
            probabilies = self.__normalize(self.vo_matrix[v_index, :])

            ## All zero probabilities
            if not np.any(probabilies):
                return np.random.choice(list(self.vo_object_dict.keys()), size=1)[0]  # uniform

            return np.random.choice(list(self.vo_object_dict.keys()), size=1, p=probabilies)[0]
        
        if V in self.vo_verb_dict:
            v_index = self.vo_verb_dict[V]
            return pred_o(v_index)
        else:
            # TODO: Do some stemming or something if V is given
            return "-1"


    
    def pred_sent(self, feature):
        word, pos = feature
        
        ## Predict POS tag for word if it wasn't given
        if pos is None:
            pos = self.pred_pos(word)

        word = word.lower()
        given_svo = self.pred_svo((word, pos))

        if given_svo == "subj":
            # TODO: Extend subject
            S = word
            if pos == 'NOUN':
                S = "the " + word
            
            ## Use co-occurence matrix and embeddings to predict V and O
            ## Use random choice to predict for unkown embedding
            V = self.__pred_sv(S)
            O = self.__pred_vo(V)

            if O is None:
                svo = (S, V)
            else:
                svo = (S, V, O)
        else:
            return given_svo
        
        sent = " ".join(svo)
        return sent.capitalize() + "."


        

        



