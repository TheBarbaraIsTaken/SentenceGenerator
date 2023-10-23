from model import Predicter

def load_corpus(file_path):
    sentences = """
        The big black cat stared at the small dog. 
        Jane watched her brother in the evenings.
        Have you seen it?
        I want a cat.
        The new car was bought by me.
    """

    return sentences

if __name__ == "__main__":
    text = load_corpus("")

    model = Predicter(text, stem=False)
    feature = ("Cat", None)

    print(model.pred_sent(feature))
