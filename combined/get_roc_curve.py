# --- Step 1: Imports ---
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import re

# --- Step 2: Load Dataset ---
df = pd.read_csv("url_dataset.csv")

# Optional: sample for speed during dev
# df = df.sample(frac=0.01, random_state=42)

# --- Step 3: Extract handcrafted features ---
def extract_url_features(df):
    df['url_length'] = df['URL'].apply(len)
    df['num_dots'] = df['URL'].apply(lambda x: x.count('.'))
    df['has_at'] = df['URL'].apply(lambda x: int('@' in x))
    df['has_https'] = df['URL'].apply(lambda x: int(urlparse(x).scheme == 'https'))
    df['has_ip'] = df['URL'].apply(lambda x: int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", urlparse(x).netloc))))
    return df

df = extract_url_features(df)

# --- Step 4: Prepare data ---
FEATURE_KEYS = ['url_length', 'num_dots', 'has_at', 'has_https', 'has_ip']
urls = df['URL'].astype(str).values
y = df['label'].values

# --- Step 5: Load tokenizer & scaler ---
tokenizer = joblib.load("url_tokenizer.pkl")
scaler = joblib.load("feature_scaler.pkl")

# --- Step 6: Preprocess inputs ---
MAX_LEN = 200

# Tokenize URLs
X_url = tokenizer.texts_to_sequences(urls)
X_url = pad_sequences(X_url, maxlen=MAX_LEN)

# Scale handcrafted features
X_feat = df[FEATURE_KEYS].values
X_feat = scaler.transform(X_feat)

# --- Step 7: Split data ---
X_url_train, X_url_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_url, X_feat, y, test_size=0.2, random_state=42
)

# --- Step 8: Load the model ---
model = load_model("phishing_combined_model.h5")

# --- Step 9: Predict probabilities ---
y_probs = model.predict([X_url_test, X_feat_test]).flatten()

# --- Step 10: Compute ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# --- Step 11: Plot ROC ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Step 12: Best threshold (optional) ---
optimal_idx = (tpr - fpr).argmax()
best_thresh = thresholds[optimal_idx]
print(f"Best threshold based on ROC: {best_thresh:.4f}")
