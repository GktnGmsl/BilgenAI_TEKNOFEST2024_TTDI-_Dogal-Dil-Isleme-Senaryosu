#utils.py
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from zeyrek import MorphAnalyzer
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
def temizle(text):
    text = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s]', '', text)
    text = re.sub(r'\d', 'NUM', text)
    stop_words = set(stopwords.words('turkish'))
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text


def find_entity(text, lstm_model, vectorizer):
    # Metni temizle
    cleaned_text = temizle(text)

    # Morphological Analyzer
    analyzer = MorphAnalyzer()

    # Entity listesini oluştur
    entity_list = []
    words = cleaned_text.split()
    i = 0

    while i < len(words):
        entity = ""
        if words[i].istitle() or words[i].isupper():
            entity = words[i]
            i += 1
            while i < len(words) and (words[i].istitle() or words[i].isupper()):
                entity += ' ' + words[i]
                i += 1

            analyzed_entity = analyzer.analyze(entity)
            if analyzed_entity:
                entity_root = analyzed_entity[0][0].lemma
                entity_list.append(entity_root)
            else:
                entity_list.append(entity)

        # Belirli eklerle biten kelimeleri entity olarak kabul et
        elif words[i].endswith('lık') or words[i].endswith('lik') or words[i].endswith('luğu') or words[i].endswith(
                'lugu'):
            entity_list.append(words[i])

        # Web adresleri ve alan adları gibi özel durumları entity olarak kabul et
        elif any(words[i].startswith(prefix) for prefix in ['www.', 'http://', 'https://']):
            entity_list.append(words[i])
        elif any(words[i].endswith(suffix) for suffix in ['.com', '.org', '.net']):
            entity_list.append(words[i])

        else:
            i += 1

    # Metni vektörize etmeye yarıyor.
    text_vector = vectorizer.transform([cleaned_text]).toarray()

    # LSTM modelinden entity tahminlerini almaya yarıyor.
    predictions = lstm_model.predict(text_vector)

    # Tahminleri işle ve UNK kontrolü yapmaya yarıyor.
    for idx, pred in enumerate(predictions[0]):
        if pred > 0.5:  # %50'den yüksekse entity olarak kabul ediyor
            entity = tokenizer.decode([idx])  # Token'dan entity'yi decode ediyor
            if entity.lower() != 'unk':  # UNK kontrolü yapıyor
                entity_list.append(entity)

    return entity_list, predictions


def encode_entities(entity_list):
    le = LabelEncoder()
    encoded_entities = le.fit_transform(entity_list)
    return encoded_entities
