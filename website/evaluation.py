from model_parser import Predicter
import os
import numpy as np
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')



if __name__ == "__main__":
    conll_file = "../Data/UD_English-EWT/en_ewt-ud-test.conllu"

    model = Predicter(conll_file)

    size = 100

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

    print("Ready")
    