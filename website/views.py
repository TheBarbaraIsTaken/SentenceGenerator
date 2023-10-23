from flask import Blueprint, render_template, request, flash
from . import predicter_model

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
            

            feature = (key_word, pos)

            return render_template("home.html", sentence=predicter_model.pred_sent(feature), show=True)


    return render_template("home.html", show=False)

