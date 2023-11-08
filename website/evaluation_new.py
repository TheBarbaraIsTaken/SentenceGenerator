from model_parser import Predicter
import os
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

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
    conll_file = os.path.abspath(r'C:\Users\tobia\Documents\UTwente\ITECH\Natural Language Processing\Project\SentenceGenerator\Data\UD_English-EWT\en_ewt-ud-test.conllu')
    file = "hp.txt"

    model = Predicter(conll_file)  # Predicter("hp.txt", stem=False)
    feature = ("it", None)
    
    
    for i in range(10):
        s = (model.pred_sent(feature))
        print(s)

        matches = tool.check(s)
        print(len(matches))