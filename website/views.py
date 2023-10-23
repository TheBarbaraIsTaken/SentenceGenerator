from flask import Blueprint, render_template, request, flash
from .model import Predicter

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # TODO: check if key_word and pos are set, key_word is requared, pos is nullable -> set it None
        key_word = request.form.get('key_word')
        pos = request.form.get('pos')

        if not pos:
            pos = None

        if not key_word:
            flash('Key word must be given', category='error')
        elif len(key_word) < 2:
            flash('Key word must be greater than 1 character', category='error')
        else:
            text = load_corpus('')
            model = Predicter(text, stem=False) ## Note: stemming is not a good idea because our sentences won't make any sense

            feature = (key_word, pos)

            return render_template("home.html", sentence=model.pred_sent(feature), show=True)


    return render_template("home.html", show=False)


def load_corpus(file_path):
    sentences = """
        The big black cat stared at the small dog. 
        Jane watched her brother in the evenings.
        Have you seen it?
        I want a cat.
        The new car was bought by me.
    """

    return sentences