from model_parser import Predicter


def load_corpus(file_path):

    if file_path:
        with open(file_path, 'r') as f:
            sentences = f.read()
    else:
        sentences = """
            I have a cat.
            She has a dog.
        """

    return sentences

if __name__ == "__main__":
    # text = load_corpus("hp.txt")
    conll_file = "../Data/UD_English-EWT/en_ewt-ud-test.conllu"
    #conll_file = "../Data/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu"
    file = "hp.txt"
    
    model = Predicter(conll_file)  # Predicter("hp.txt", stem=False)
    feature = ("table", None)
    
    
    for i in range(10):
        s = (model.pred_sent(feature))
        print(s)

