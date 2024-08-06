import joblib
import pandas as pd
from zeyrek import MorphAnalyzer
import nltk
from flask import Flask, render_template, request,url_for

nltk.download('punkt')

# Modeli yükle
hukukBolumleri = joblib.load('hukukBolumleri.joblib')

# Zeyrek kütüphanesini kullanarak Türkçe için morfolojik analizci yeniden yükle
analyzer = MorphAnalyzer()

app = Flask(__name__, template_folder='templates', static_folder='templates/statics')

# Kök Bulma Fonksiyonu
def find_root(kelime):
    analysis = analyzer.lemmatize(kelime)
    return analysis[0][1][0] if analysis else kelime

# Metin Analiz Fonksiyonu
def analyze_text(metin):
    metin = metin.lower().split()
    metin = [find_root(kelime) for kelime in metin]
    kelimeSayilari = {sütun: 0 for sütun in hukukBolumleri.columns}

    for kelime in metin:
        for sütun in hukukBolumleri.columns:
            if kelime in hukukBolumleri[sütun].values:
                kelimeSayilari[sütun] += 1

    ilgiliBolum = max(kelimeSayilari, key=kelimeSayilari.get)
    return ilgiliBolum

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        problem_text = request.form['problemText']
        result = analyze_text(problem_text)
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
