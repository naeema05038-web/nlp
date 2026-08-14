import joblib
import os

# 1. Get the exact folder path automatically
current_file_path = os.path.abspath(__file__)
src_folder = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_folder)

# 2. Build the full paths to the models (NOTE: folder is "model", not "models")
model_path = os.path.join(project_root, "model", "mental_health_model.pkl")
vectorizer_path = os.path.join(project_root, "model", "tfidf_vectorizer.pkl")

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("❌ Error: Model or Vectorizer files not found.")
    print(f"Checked path for Model: {model_path}")
    print(f"Checked path for Vectorizer: {vectorizer_path}")
    print("\nPlease check that your 'model' folder is right next to your 'src' folder.")
else:
    # Load the model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("✅ Mental Health model loaded successfully!")
    print("✅ Vectorizer loaded successfully!")
    print("-" * 40)

    # 2. Define the testing function
    def predict_mental_health(user_sentence):
        test_vector = vectorizer.transform([user_sentence])
        prediction = model.predict(test_vector)
        return prediction[0]

    # 3. Run the Real Sentence Prediction Tests
    test_sentences = [
        "I feel worried and nervous all the time.",
        "I want to end everything, I can't do this anymore.",
        "I am feeling very frustrated with my life.",
        "I'm just tired of waking up every single day."
    ]

    print("🔎 Testing Real Sentences:\n")
    for sentence in test_sentences:
        label = predict_mental_health(sentence)
        print(f"Input: \"{sentence}\"")
        print(f"Predicted Label: {label}")
        print("-" * 40)