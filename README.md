# LawNav ⚖️

LawNav, kullanıcıların yazdığı hukuki sorun metinlerini analiz ederek bunların hangi hukuk alanıyla ilgili olabileceğini tahmin eden web tabanlı bir uygulamadır.

Proje, Türkçe metin işleme teknikleri kullanılarak geliştirilmiştir ve uçtan uca bir yapı sunar: kullanıcı arayüzü, metin ön işleme, analiz ve sonuç gösterimi.

## Akademik çalışma

Projenin teorik altyapısı ve yöntemleri akademik bir çalışmada ele alınmıştır:

👉 [IECSR 2024 (Zurich) Akademik Bildirisi / Raporu](https://www.researchgate.net/publication/384500931_LAWNAV_YAPAY_ZEKA_DESTEKLI_HUKUK_DANISMANI_YONLENDIRME_UYGULAMASI)

## Özellikler

- Türkçe metinleri işlemek için lemmatization kullanır.
- Hukuki metni analiz ederek ilgili hukuk alanını tahmin eder.
- Flask tabanlı bir backend yapısına sahiptir.
- Basit ve kullanıcı dostu bir web arayüzü sunar.
- Kullanıcı girişini destekleyen karakter sayacı bulunur.

## Kullanılan teknolojiler

- Backend: Python, Flask
- NLP: Zeyrek, NLTK, Pandas, Joblib
- Frontend: HTML5, CSS3, JavaScript

## Nasıl çalışır

1. Kullanıcı hukuki sorununu metin olarak girer.
2. Metin ön işleme aşamasından geçirilir.
3. Kelimeler köklerine indirgenir.
4. İşlenmiş veri, kayıtlı model/veri matrisi ile karşılaştırılır.
5. Sistem en olası hukuk alanını tahmin ederek sonucu gösterir.

## Kurulum

```bash
pip install -r requirements.txt
python app.py
```

Ardından tarayıcıda şu adresi açın:

```text
http://localhost:5000
```

## Proje yapısı

```text
├── app.py
├── requirements.txt
├── hukukBolumleri.joblib
└── templates/
    ├── index.html
    ├── result.html
    └── statics/
        ├── style.css
        └── theme.js
```

## Not

Bu proje hukuki danışmanlık vermek için değil, kullanıcıyı ilgili hukuk alanına yönlendirmeye yardımcı olmak için geliştirilmiştir.
