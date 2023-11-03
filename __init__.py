from flask import Flask
from .model_parser import Predicter
import os


def load_corpus(file_path):
    if file_path:
        path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(path, file_path), 'r') as f:
            sentences = f.read()
    else:
        sentences = """
            The big black cat stared at the small dog. 
            Jane watched her brother in the evenings.
            Have you seen it?
            I want a cat.
            The new car was bought by me.
        """

    return sentences

text = load_corpus('') # hp.txt
#predicter_model = Predicter(text, stem=False) ## Note: stemming is not a good idea because our sentences won't make any sense
predicter_model = Predicter("../Data/UD_English-EWT/en_ewt-ud-test.conllu")

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'lo9i8765r4edcvbghji9oekpldkmjh'

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
