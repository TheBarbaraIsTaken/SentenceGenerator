from flask import Blueprint, render_template, request, flash
from . import predicter_model

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        key_word = request.form.get('key_word')
        pos = request.form.get('pos')

        # POS: nullable
        if not pos or pos == "NONE":
            pos = None

        print(pos)

        """
        # Key word: nullable
        if not key_word:
            flash('Key word must be given', category='error')
        elif len(key_word) < 2:
            flash('Key word must be greater than 1 character', category='error')
        else:
            feature = (key_word, pos)

            return render_template("home.html", show=True, sentence=predicter_model.pred_sent(feature))"""


    return render_template("home.html", show=False)

