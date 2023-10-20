from model import Predicter

if __name__ == "__main__":
    sentences = """
        The big black cat stared at the small dog. 
        Jane watched her brother in the evenings.
        Have you seen it?
        I want a cat.
        The new car was bought by me.
    """

    feature = ("cat", None)

    model = Predicter(sentences)

    print(model.pred_sent(feature))
