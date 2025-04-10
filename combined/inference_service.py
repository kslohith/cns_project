from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse
import re


#parse url
def has_ip_in_domain(url):
    try:
        domain = urlparse(url).netloc
        return int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain)))
    except:
        return 0

#feature extractor from url
def extract_url_features(url):
    parsed = urlparse(url)

    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_at': int('@' in url),
        'has_https': int(parsed.scheme == 'https'),
        'has_ip': has_ip_in_domain(url)
    }

# Load model, tokenizer, scaler
model = tf.keras.models.load_model("phishing_combined_model.h5")
tokenizer = joblib.load("url_tokenizer.pkl")
scaler = joblib.load("feature_scaler.pkl")

# Settings
MAX_LEN = 200  # Same as used in training
FEATURE_KEYS = ['url_length', 'num_dots', 'has_at', 'has_https', 'has_ip']

# Inference function
def predict_url(url: str) -> int:
    # Tokenize URL
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Extract and scale handcrafted features
    feats = extract_url_features(url)
    feat_vector = scaler.transform([list(feats[key] for key in FEATURE_KEYS)])

    # Predict
    pred = model.predict([padded, feat_vector])[0][0]

    print(pred)

    # best threshold value according to the roc curve
    return int(pred <= 0.0218)

# Set up Flask app
app = Flask(__name__)

@app.route("/predict-url", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        print(url)
        prediction = predict_url(url)
        return jsonify({"url": url, "phishing": bool(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
