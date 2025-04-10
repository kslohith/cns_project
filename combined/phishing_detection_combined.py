import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, GlobalMaxPooling1D, Dense, Input, Concatenate, Conv1D, Dropout, Bidirectional, LSTM 

# Extract handcrafted features
def extract_url_features(df):
    df['url_length'] = df['URL'].apply(len)
    df['num_dots'] = df['URL'].apply(lambda x: x.count('.'))
    df['has_at'] = df['URL'].apply(lambda x: int('@' in x))
    df['has_https'] = df['URL'].apply(lambda x: int('https' in x))
    df['has_ip'] = df['URL'].apply(lambda x: int(any(char.isdigit() for char in x.split('/')[2]) if '//' in x else 0))
    return df

# load data and extract features
df = pd.read_csv("url_dataset.csv")
df = extract_url_features(df)

# Prepare data
urls = df['URL'].astype(str).values
labels = df['label'].values

# Character-level tokenization
tokenizer = Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(urls)
sequences = tokenizer.texts_to_sequences(urls)

# Pad sequences
max_len = 200  # Adjust based on URL length distribution
X_url = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
feature_cols = ['url_length', 'num_dots', 'has_at', 'has_https', 'has_ip']
X_handcrafted = df[feature_cols].values
scaler = StandardScaler()
X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)
y = labels

# Train/test split
X_url_train, X_url_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_url, X_handcrafted_scaled, y, test_size=0.2, random_state=42
)

# train the neural network

url_input = Input(shape=(max_len,), name='url_input')
feat_input = Input(shape=(X_handcrafted_scaled.shape[1],), name='meta_input')


# Embedding branch
vocab_size = len(tokenizer.word_index) + 1
x = Embedding(input_dim=vocab_size, output_dim=64)(url_input)
x = Bidirectional(LSTM(64))(x)

# Concatenate
x = Concatenate()([x, feat_input])
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Compile
model = Model(inputs=[url_input, feat_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit([X_url_train, X_feat_train], y_train, epochs=10, batch_size=32, validation_split=0.2)


# Testing the model
loss, acc = model.evaluate([X_url_test, X_feat_test], y_test)
print(f"Test Accuracy: {acc:.4f}")

# saving model details
model.save("phishing_combined_model.h5")
joblib.dump(tokenizer, "url_tokenizer.pkl")
joblib.dump(scaler, "feature_scaler.pkl")