# Customer Category Purchase Prediction

Bu proje, müşterilerin geçmiş satın alma davranışlarına dayanarak yeni ürün kategorilerinde satın alma olasılıklarını tahmin eden bir derin öğrenme modeli içerir.

## Proje Yapısı

```
customer_categorical_order_prediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   └── model_evaluation.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

## Kurulum

1. Sanal ortam oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. `.env.example` dosyasını `.env` olarak kopyalayın ve veritabanı bağlantı bilgilerinizi girin.

## Kullanım

1. Veri hazırlama:
```bash
python src/data/feature_engineering.py
```

2. Model eğitimi:
```bash
python src/models/neural_network.py
```

## Test

```bash
pytest tests/
```

## Kod Kalitesi

- Black ile kod formatlaması
- Flake8 ile kod analizi
- MyPy ile tip kontrolü
- Pytest ile birim testleri 