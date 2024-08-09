# nlp.py
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel

import utils
from utils import tokenizer

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
model = BertModel.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
#BERTurk-128k-cased kullanılarak veri tokenize ediliyor.
def tokenize_text(text):
    inputs = tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return inputs
#Sentiment Analizi attention mask'e maruz kalarak tahmin edilen cevapların değerlerini ayarlıyor.
def sentiment_analysis(text):
    inputs = tokenize_text(text)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model(input_ids, attention_mask)
    _, predicted = torch.max(outputs.last_hidden_state[:, 0, :], dim=1)
    return predicted.item()
#CBOW yapısı tanımlanır.
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        return embeddings

cbow_model = CBOW(len(tokenizer.vocab), embedding_dim=128)

def load_lstm_model():
    return tf.keras.models.load_model('lstm_model.h5')
#Eğitilen LSTM NER modelini çağırır.
def ner(text, vectorizer):
    lstm_model = load_lstm_model()
    entity_list, lstm_preds = utils.find_entity(text, lstm_model, vectorizer)

    # 'Unk' olanları filtreler
    filtered_entity_list = [entity for entity in entity_list if entity.lower() != 'unk']

    return filtered_entity_list, lstm_preds
