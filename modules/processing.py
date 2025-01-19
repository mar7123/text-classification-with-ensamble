import ast
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder


def join_text_list(texts):
    texts = ast.literal_eval(str(texts))
    return ' '.join([text for text in texts])


def compute_tfidf_features(data, max_features):
    # Initialize TF-IDF vectorizer
    model_name = "training\\tfidf_vectorizer.pkl"
    model_path = os.path.join(os.getcwd(), model_name)
    tfidf_vectorizer = joblib.load(model_path)

    sorted_vocab = sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x: x[1])[:max_features]
    truncated_vocab = {term: idx for idx, (term, _) in enumerate(sorted_vocab)}

    tfidf_vectorizer_adjusted = TfidfVectorizer(
        max_features=max_features,
        vocabulary=truncated_vocab  # Use the same vocabulary
    )
    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer_adjusted.fit_transform(data).toarray()
    terms = tfidf_vectorizer_adjusted.get_feature_names_out()
    return tfidf_matrix, terms


def processing():
    pd.set_option('display.max_colwidth', None)

    preprocessed_dataset_name = "temp\\preprocessed_dataset.xlsx"
    preprocessed_dataset_path = os.path.join(os.getcwd(), preprocessed_dataset_name)

    df = pd.read_excel(preprocessed_dataset_path, usecols=['stemming'])
    df.columns = ['text']

    df["text_join"] = df["text"].apply(join_text_list)

    k = 1000  # Number of features to select
    tfidf_mat, terms = compute_tfidf_features(df["text_join"], k)
    print(tfidf_mat.shape)

    label_name = "training\\label.xlsx"
    label_path = os.path.join(os.getcwd(), label_name)

    df_label = pd.read_excel(label_path, usecols=['label'])

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df_label['label'])
    target_names = [name.replace(" ", "_") for name in label_encoder.classes_]

    if tfidf_mat.shape[1] < k:
        padded_features = np.zeros((1, k))
        padded_features[:, :tfidf_mat.shape[1]] = tfidf_mat
        model_name = "training\\stacking_classifier.pkl"
        model_path = os.path.join(os.getcwd(), model_name)

        with open(model_path, 'rb') as file:
            model = joblib.load(file)
            result = model.predict(padded_features)
            print(result)
            pred = result[0]
            return target_names[pred]

    model_name = "training\\stacking_classifier.pkl"
    model_path = os.path.join(os.getcwd(), model_name)

    with open(model_path, 'rb') as file:
        model = joblib.load(file)
        result = model.predict(tfidf_mat)
        pred = result[0]
        return target_names[pred]
