import joblib
import pandas as pd
from zeyrek import MorphAnalyzer
import nltk
from flask import Flask, render_template, request, url_for
import os

nltk.download('punkt', quiet=True)

# Mutlak yol tanımla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modeli yükle
hukukBolumleri = joblib.load(os.path.join(BASE_DIR, 'hukukBolumleri.joblib'))

analyzer = MorphAnalyzer()

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'templates', 'statics'))

# Kök Bulma Fonksiyonu
def find_root(kelime):
    analysis = analyzer.lemmatize(kelime)
    return analysis[0][1][0] if analysis else kelime

# Metin Analiz Fonksiyonu
def analyze_text(metin):
    if not metin.strip():
        return "Metin boş!"
    
    metin = metin.lower().split()
    metin = [find_root(kelime) for kelime in metin]
    kelimeSayilari = {sutun: 0 for sutun in hukukBolumleri.columns}

    for kelime in metin:
        for sutun in hukukBolumleri.columns:
            if kelime in hukukBolumleri[sutun].values:
                kelimeSayilari[sutun] += 1

    if max(kelimeSayilari.values()) == 0:
        return "İlgili hukuk bölümü bulunamadı."

    ilgiliBolum = max(kelimeSayilari, key=kelimeSayilari.get)
    return ilgiliBolum

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    problem_text = request.form['problemText']
    result = analyze_text(problem_text)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)