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
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features, smooth_idf=True, norm='l2', stop_words=None
    )
    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data).toarray()
    terms = tfidf_vectorizer.get_feature_names_out()

    return tfidf_matrix, terms


def processing():
    pd.set_option('display.max_colwidth', None)

    preprocessed_dataset_name = "temp\\preprocessed_dataset.xlsx"
    preprocessed_dataset_path = os.path.join(os.getcwd(), preprocessed_dataset_name)

    df = pd.read_excel(preprocessed_dataset_path, usecols=['stemming'])
    df.columns = ['text']

    df["text_join"] = df["text"].apply(join_text_list)

    max_features = 1000

    tfidf_mat, terms = compute_tfidf_features(df["text_join"], max_features)
    print(tfidf_mat.shape)

    label_name = "training\\label.xlsx"
    label_path = os.path.join(os.getcwd(), label_name)

    df_label = pd.read_excel(label_path, usecols=['label'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_label['label'])
    target_names = [name.replace(" ", "_") for name in label_encoder.classes_]


    k = 500  # Number of features to select
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



    def apply_chi2_feature_selection(X, y, k_inner=500):
        chi2_selector = SelectKBest(chi2, k=k_inner)
        X_kbest = chi2_selector.fit_transform(X, y)
        mask = chi2_selector.get_support()
        selected_features = [terms[i] for i in range(len(mask)) if mask[i]]
        return X_kbest, selected_features

    X_selected, selected_terms = apply_chi2_feature_selection(tfidf_mat, y_encoded, k_inner=k)

    model_name = "training\\stacking_classifier.pkl"
    model_path = os.path.join(os.getcwd(), model_name)

    with open(model_path, 'rb') as file:
        model = joblib.load(file)
        result = model.predict(X_selected)
        pred = result[0]
        return target_names[pred]
