# train.py
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import temizle

layers = tf.keras.layers

# Veri setini yüklenir
def load_data():
    print("Model 1.")
    df1 = pd.read_csv('C:/Users/gktng/PycharmProjects/pythonProject4/sikayet-var-full.csv')
    df2 = pd.read_csv('C:/Users/gktng/PycharmProjects/pythonProject5/Pozitifsikayet-var-full.csv')
    df1 = df1.sample(frac=0.8)  # Negatif veri setinden %80 oranında örnek alınır
    df2 = df2.sample(frac=0.2)  # Pozitif veri setinden %20 oranında örnek alınır
    combined_df = pd.concat([df1, df2], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)  # Veri setini karıştırır
    return shuffled_df

# Etiketleme sisteminin bulunduğu fonksiyon
def label_data(df):
    print("Model 2.")
    etiketler = []
    for text in df['description']:
        if 'olumsuz' in text.lower():
            etiketler.append(0)  # Olumsuz sınıf etiketi 0 olarak atanır
        elif 'nötr' in text.lower():
            etiketler.append(2)  # Nötr sınıf etiketi 2 olarak atanır
        else:
            etiketler.append(1)  # Olumlu sınıf etiketi 1 olarak atanır
    df['label'] = etiketler
    return df

# Metni vektör haline getirir
def vectorize_text(df):
    print("Model 3.")
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors = vectorizer.fit_transform(df['description'])
    return vectors.toarray()

# Random Forest Modelini eğitir
def train_model(X_train, y_train):
    try:
        print("Model 4.")
        smote = ADASYN(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )

        rf_model.fit(X_train_res, y_train_res)
        return rf_model
    except Exception as e:
        print(f"Hata: {e}")
        return None

# LSTM Modeli eğitir.
def train_lstm_model(X_train, y_train, X_test, y_test):
    print("Model 5.")
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=10000, output_dim=64, input_length=X_train.shape[1]),
        layers.LSTM(units=32, dropout=0.2),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model

# Threshold ayarını ayarlar ve sınıflandırma sınırını ayarlar.
def adjust_threshold(y_pred_proba, threshold=0.5):
    return (y_pred_proba[:, 1] >= threshold).astype(int)

def train_and_evaluate(df):
    print("Model 6.")
    df = label_data(df)
    df['description'] = df['description'].apply(temizle)
    X_train, X_test, y_train, y_test = train_test_split(df['description'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test)
    X_train_count = X_train_count.toarray()
    X_test_count = X_test_count.toarray()

    model = train_model(X_train_count, y_train)
    if model is not None:
        y_pred_proba = model.predict_proba(X_test_count)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]  # Various thresholds to try

        for threshold in thresholds:
            y_pred = adjust_threshold(y_pred_proba, threshold=threshold)
            print(f"Threshold: {threshold}")
            print(f"Doğruluk: {accuracy_score(y_test, y_pred)}")
            print(f"AUC-ROC: {roc_auc_score(y_test, y_pred, multi_class='ovr')}")
            print(classification_report(y_test, y_pred))

    lstm_model = train_lstm_model(X_train_count, y_train, X_test_count, y_test)
    return model, lstm_model

# Modeli kaydeder
def save_model(model, lstm_model):
    print("Model 7.")
    import pickle
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    lstm_model.save('lstm_model.h5')


if __name__ == "__main__":
    print("Model 8.")
    df = load_data()
    model, lstm_model = train_and_evaluate(df)
    save_model(model, lstm_model)
