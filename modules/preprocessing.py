import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

nltk.download('punkt_tab')
nltk.download('stopwords')

def case_folding(text):
    return text.lower()


def remove_number(text):
    return re.sub(r"\d+", "", text)


def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)


def remove_whitespace_LT(text):
    return text.strip()


def remove_whitespace_multiple(text):
    return re.sub(r'\s+', ' ', text)


def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


def word_tokenize_wrapper(text):
    return word_tokenize(text)

def stemmed_wrapper(term, stemmer):
    return stemmer.stem(term)


def preprocess(dataset_path):
    pd.set_option('display.max_colwidth', None)

    df = pd.read_excel(dataset_path)

    df.head()

    print('Dataset size:', df.shape)
    print('Columns are:', df.columns)

    df.head()

    df['cleaning'] = df['response'].apply(remove_number)
    df['cleaning'] = df['cleaning'].apply(remove_punctuation)
    df['cleaning'] = df['cleaning'].apply(remove_whitespace_LT)
    df['cleaning'] = df['cleaning'].apply(remove_whitespace_multiple)
    df['cleaning'] = df['cleaning'].apply(remove_single_char)

    df['case_folding'] = df['cleaning'].apply(case_folding)

    df.head()

    df['tokenizing'] = df['cleaning'].apply(word_tokenize_wrapper)

    df.head()

    list_stopwords = stopwords.words('indonesian')
    len(list_stopwords)

    stopwords_name = "training\\stopwords.txt"
    stopwords_path = os.path.join(os.getcwd(), stopwords_name)
    txt_stopword = pd.read_csv(stopwords_path, names=["stopwords"], header=None)

    # convert stopword string to list & append additional stopword
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    len(list_stopwords)

    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['filtering'] = df['tokenizing'].apply(stopwords_removal)

    df.head()

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    term_dict = {}

    for document in df['filtering']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term, stemmer)
        print(term, ":", term_dict[term])

    print(term_dict)
    print("------------------------")

    term_dict_name = "training\\term_dict_strict.pkl"
    term_dict_path = os.path.join(os.getcwd(), term_dict_name)

    with open(term_dict_path, 'wb') as f:
        pickle.dump(term_dict, f)

    def get_stemmed_term(term_document):
        return [term_dict[term_doc] for term_doc in term_document]

    df['stemming'] = df['filtering'].apply(get_stemmed_term)

    df.head()

    preprocessed_dataset_name = "temp\\preprocessed_dataset.xlsx"
    preprocessed_dataset_path = os.path.join(os.getcwd(), preprocessed_dataset_name)
    df.to_excel(preprocessed_dataset_path, index=False)
    return df