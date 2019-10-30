from flask import Flask
from flask import request, jsonify
import base64
import pickle

# загружаем модель и векторайзер из ДЗ №10
lsvc = pickle.load(open('lsvc.pckl', 'rb'))
tfidf_w2v = pickle.load(open('tfidf_w2v.pckl', 'rb'))

app = Flask(__name__)

@app.route("/predict",  methods=['POST'])
def hello():
    text = request.form.get('text')
    resp = {
        'predict': int(lsvc.predict(tfidf_w2v.transform([text]))[0]),
        # возвращаем предсказанный номер класса (от 0 до 5)
    }
    return jsonify(resp)

app.run(host='0.0.0.0', port=11018, debug=True)

# запускается командой из консоли cd /14_reallife/flask_hw11.py
# протестировать, запустив requests_for_hw11.py или hw11_flask_w2v.ipynb
