import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_dataset(df):
    # Remove missing text
    df = df.dropna(subset=["text"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def prepare_text(df):
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_tfidf(X_train, X_test):
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf