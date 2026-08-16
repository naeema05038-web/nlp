import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "emotion_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "emotion_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict_emotion(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)

    return prediction[0]


if __name__ == "__main__":
    test_text = "I am very happy and excited today!"
    result = predict_emotion(test_text)

    print("Text:", test_text)
    print("Predicted Emotion:", result)