from model_parser import Predicter
import os
import numpy as np
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def write_sentences_to_file(size=100):
    word_pos = [(word, pos) for word, pos in model.vocab_hmm.keys() if pos is not None and pos in model.hmm_pos]
    random_inx = np.random.choice(len(word_pos), replace=False, size=size)
    random_features = [word_pos[i] for i in random_inx]
    sentences = []

    for feature in random_features:
        print(feature)
        sentence = model.pred_sent(feature)
        sentences.append(sentence)
    
    print("==========================================================")

    errors = []

    ## Write sentences to file
    with open("../Data/out/result.txt", "w") as f:
        for sentence in sentences:
            matches = tool.check(sentence)
            error_num = len(matches)

            errors.append(error_num)
            print(sentence, error_num, file=f, sep="\t")

    ## Save errors
    errors = np.asarray(errors)
    np.save('../Data/out/errors.npy', errors)

def evaluate_errors(times=100, size=100):
    ## Get random features
    word_pos = [(word, pos) for word, pos in model.vocab_hmm.keys() if pos is not None and pos in model.hmm_pos]
    random_inx = np.random.choice(len(word_pos), replace=False, size=size)
    random_features = [word_pos[i] for i in random_inx]
    
    all_score = []

    f = open("../Data/out/sentences.txt", "w")

    for _ in range(times):
        scores = []

        for feature in random_features:
            sentence = model.pred_sent(feature)
            print(sentence, file=f)
            matches = tool.check(sentence)
            error_num = len(matches)

            scores.append(error_num)

        all_score.append(scores)

    f.close()

    with open("../Data/out/features.txt", "w") as f:
        for feature in random_features:
            print(feature, file=f)

    return np.asarray(all_score)


if __name__ == "__main__":
    conll_file = "../Data/UD_English-EWT/en_ewt-ud-test.conllu"

    model = Predicter(conll_file)

    scores = evaluate_errors()
    np.save('../Data/out/all_errors.npy', scores)
    

    print("Ready")
    