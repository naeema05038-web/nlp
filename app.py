import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MindInsight | AI Mental Health Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# MODEL PATHS
# =========================================================

EMOTION_MODEL_PATH = "models/emotion_model.pkl"
EMOTION_VECTORIZER_PATH = "models/emotion_vectorizer.pkl"

MENTAL_HEALTH_MODEL_PATH = "models/mental_health_model.pkl"
MENTAL_HEALTH_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# =========================================================
# LOAD MODELS
# =========================================================

models_loaded = True

try:

    emotion_model = joblib.load(
        EMOTION_MODEL_PATH
    )

    emotion_vectorizer = joblib.load(
        EMOTION_VECTORIZER_PATH
    )

    mental_health_model = joblib.load(
        MENTAL_HEALTH_MODEL_PATH
    )

    mental_health_vectorizer = joblib.load(
        MENTAL_HEALTH_VECTORIZER_PATH
    )

except Exception as e:

    models_loaded = False

    emotion_model = None
    emotion_vectorizer = None
    mental_health_model = None
    mental_health_vectorizer = None

    model_error = str(e)


# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 15% 5%,
                rgba(99, 76, 190, 0.16),
                transparent 30%
            ),

            radial-gradient(
                circle at 90% 10%,
                rgba(35, 150, 150, 0.08),
                transparent 28%
            ),

            #080b12;

        color: #f3f5fa;
    }


    .block-container {

        max-width: 1050px;

        padding-top: 2rem;

        padding-bottom: 2rem;
    }


    /* =====================================================
       TEXT
       ===================================================== */

    h1 {

        color: #ffffff !important;

        font-weight: 800 !important;
    }


    h2 {

        color: #e8eaff !important;

        font-weight: 750 !important;
    }


    h3 {

        color: #e3e6ef !important;

        font-weight: 700 !important;
    }


    p {

        color: #9da6b7;
    }


    /* =====================================================
       CAPTION
       ===================================================== */

    [data-testid="stCaptionContainer"] {

        color: #7f899b !important;
    }


    /* =====================================================
       SYSTEM READY
       ===================================================== */

    [data-testid="stAlert"] {

        border-radius: 10px !important;

        border: 1px solid rgba(45, 212, 191, 0.25) !important;

        background: rgba(20, 184, 166, 0.10) !important;

        color: #8ee9dc !important;
    }


    /* =====================================================
       TEXT AREA
       ===================================================== */

    textarea {

        background-color: #0d121c !important;

        color: #f7f9ff !important;

        border: 1px solid #293449 !important;

        border-radius: 14px !important;

        font-size: 15px !important;

        line-height: 1.6 !important;
    }


    textarea:hover {

        border-color: #3b4964 !important;
    }


    textarea:focus {

        border-color: #6d5cff !important;

        box-shadow:
            0 0 0 1px #6d5cff,
            0 0 18px rgba(109,92,255,0.15) !important;
    }


    /* =====================================================
       ALL BUTTONS
       ===================================================== */

    .stButton > button {

        width: 100%;

        min-height: 44px;

        border-radius: 11px !important;

        border: 1px solid #303b52 !important;

        background: #121927 !important;

        color: #dce2ee !important;

        font-size: 14px !important;

        font-weight: 700 !important;

        transition: all 0.2s ease !important;
    }


    .stButton > button:hover {

        background: #1a2234 !important;

        border-color: #6759d8 !important;

        color: #ffffff !important;

        transform: translateY(-1px);

        box-shadow:
            0 7px 20px rgba(86,72,190,0.20);
    }


    /* =====================================================
       ANALYZE TEXT BUTTON
       ===================================================== */

    button[kind="primary"] {

        background:
            linear-gradient(
                135deg,
                #14b8a6,
                #22d3ee
            ) !important;

        border: 1px solid #2dd4bf !important;

        color: #000000 !important;

        min-height: 52px !important;

        font-size: 18px !important;

        font-weight: 800 !important;

        letter-spacing: 0.3px;

        box-shadow:
            0 8px 25px rgba(20,184,166,0.25);
    }


    button[kind="primary"] p {

        color: #000000 !important;

        font-size: 18px !important;

        font-weight: 800 !important;
    }


    button[kind="primary"]:hover {

        background:
            linear-gradient(
                135deg,
                #2dd4bf,
                #67e8f9
            ) !important;

        border-color: #5eead4 !important;

        color: #000000 !important;

        transform: translateY(-2px);

        box-shadow:
            0 12px 30px rgba(20,184,166,0.35);
    }


    button[kind="primary"]:hover p {

        color: #000000 !important;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                #121927,
                #0d131e
            );

        border: 1px solid #263149;

        border-radius: 16px;

        padding: 18px;

        min-height: 130px;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.18);
    }


    [data-testid="stMetricLabel"] {

        color: #8f99ab !important;
    }


    [data-testid="stMetricValue"] {

        color: #f7f8ff !important;

        font-weight: 800 !important;
    }


    /* =====================================================
       PROGRESS BAR
       ===================================================== */

    [data-testid="stProgressBar"] {

        background-color: #1b2332 !important;

        border-radius: 20px !important;
    }


    [data-testid="stProgressBar"] > div {

        background:
            linear-gradient(
                90deg,
                #6652e8,
                #3978e8
            ) !important;

        border-radius: 20px !important;
    }


    /* =====================================================
       INFO BOX
       ===================================================== */

    [data-testid="stInfo"] {

        background: #111d31 !important;

        border: 1px solid #243957 !important;

        border-radius: 10px !important;

        color: #9fb1c9 !important;
    }


    /* =====================================================
       WARNING
       ===================================================== */

    [data-testid="stWarning"] {

        background: rgba(130,100,20,0.12) !important;

        border: 1px solid rgba(211,177,53,0.25) !important;

        border-radius: 12px !important;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {

        border-color: #202838 !important;
    }


    /* =====================================================
       HIDE STREAMLIT UI
       ===================================================== */

    #MainMenu {

        visibility: hidden;
    }


    footer {

        visibility: hidden;
    }


    header {

        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "user_text" not in st.session_state:

    st.session_state.user_text = ""


if "show_result" not in st.session_state:

    st.session_state.show_result = False


if "emotion" not in st.session_state:

    st.session_state.emotion = ""


if "emotion_confidence" not in st.session_state:

    st.session_state.emotion_confidence = 0.0


if "mental_health" not in st.session_state:

    st.session_state.mental_health = ""


if "mental_health_confidence" not in st.session_state:

    st.session_state.mental_health_confidence = 0.0

if "emotion_probabilities" not in st.session_state:
    st.session_state.emotion_probabilities = {}

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "compare_text_a" not in st.session_state:
    st.session_state.compare_text_a = ""

if "compare_text_b" not in st.session_state:
    st.session_state.compare_text_b = ""


# =========================================================
# ADVANCED ANALYSIS HELPERS
# =========================================================

def get_probability_dict(model, vectorizer, text):
    """Return class -> probability when the model supports predict_proba."""
    try:
        vector = vectorizer.transform([text])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                return {
                    str(cls): float(prob) * 100
                    for cls, prob in zip(classes, probs)
                }
    except Exception:
        pass
    return {}


def get_text_insights(text):
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_word_length = (
        sum(len(w) for w in words) / len(words) if words else 0
    )

    positive_words = {
        "happy", "joy", "good", "great", "love", "excited",
        "hopeful", "calm", "peaceful", "proud", "wonderful"
    }
    negative_words = {
        "sad", "hopeless", "lonely", "tired", "exhausted",
        "angry", "fear", "afraid", "worried", "anxious",
        "depressed", "pain", "hate", "worthless"
    }

    lower_words = [w.lower() for w in words]
    positive_count = sum(w in positive_words for w in lower_words)
    negative_count = sum(w in negative_words for w in lower_words)

    if negative_count >= 4:
        intensity = "High"
    elif negative_count >= 2 or positive_count >= 2:
        intensity = "Moderate"
    else:
        intensity = "Low"

    return {
        "Words": len(words),
        "Sentences": len(sentences),
        "Characters": len(text),
        "Avg. word length": round(avg_word_length, 1),
        "Positive indicators": positive_count,
        "Negative indicators": negative_count,
        "Emotional intensity": intensity,
    }


def explain_linear_prediction(model, vectorizer, text, top_n=6):
    """
    Explain a linear TF-IDF prediction using the model coefficients.
    If the loaded model does not expose coef_, return a safe fallback.
    """
    try:
        if not hasattr(model, "coef_"):
            return []

        vector = vectorizer.transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        nonzero = vector.nonzero()[1]

        if len(nonzero) == 0:
            return []

        predicted = model.predict(vector)[0]
        classes = list(getattr(model, "classes_", []))

        if len(classes) == 2 and model.coef_.shape[0] == 1:
            # For binary linear classifiers, coef_ represents classes_[1].
            row = model.coef_[0]
            direction = 1 if predicted == classes[1] else -1
            scores = vector.toarray()[0] * row * direction
        else:
            class_index = classes.index(predicted)
            row = model.coef_[class_index]
            scores = vector.toarray()[0] * row

        ranked = sorted(
            [(feature_names[i], float(scores[i])) for i in nonzero],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return [
            (word, score)
            for word, score in ranked[:top_n]
            if abs(score) > 0
        ]
    except Exception:
        return []


def confidence_label(confidence):
    if confidence >= 80:
        return "High Confidence"
    if confidence >= 60:
        return "Moderate Confidence"
    return "Low Confidence"


def run_models(text):
    """Run both existing models without changing the model files."""
    emotion_vector = emotion_vectorizer.transform([text])
    emotion_prediction = emotion_model.predict(emotion_vector)[0]
    emotion = str(emotion_prediction)

    emotion_confidence = 0.0
    emotion_probabilities = {}
    if hasattr(emotion_model, "predict_proba"):
        try:
            probs = emotion_model.predict_proba(emotion_vector)[0]
            classes = getattr(emotion_model, "classes_", [])
            emotion_probabilities = {
                str(cls): float(prob) * 100
                for cls, prob in zip(classes, probs)
            }
            emotion_confidence = max(emotion_probabilities.values(), default=0.0)
        except Exception:
            pass

    mental_vector = mental_health_vectorizer.transform([text])
    mental_prediction = mental_health_model.predict(mental_vector)[0]
    mental_health = str(mental_prediction)

    mental_health_confidence = 0.0
    if hasattr(mental_health_model, "predict_proba"):
        try:
            mental_probs = mental_health_model.predict_proba(mental_vector)[0]
            mental_health_confidence = float(np.max(mental_probs)) * 100
        except Exception:
            pass

    return {
        "emotion": emotion,
        "emotion_confidence": emotion_confidence,
        "emotion_probabilities": emotion_probabilities,
        "mental_health": mental_health,
        "mental_health_confidence": mental_health_confidence,
    }


# =========================================================
# HEADER
# =========================================================

st.title("MindInsight")

st.caption(
    "AI • NLP • Text Intelligence"
)

st.success(
    "System Ready"
)

st.divider()


# =========================================================
# HERO
# =========================================================

st.markdown(
    "## Understand the Emotion Behind the Words"
)

st.write(
    "MindInsight analyzes written language using Natural Language "
    "Processing and Machine Learning to identify emotional patterns "
    "and mental-health-related text categories."
)

st.info(
    "NLP-powered text analysis • Educational project"
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    "### Tell us how you feel"
)

st.caption(
    "Write a sentence or a few lines about how you are feeling."
)


# =========================================================
# TEXT INPUT
# =========================================================

user_text = st.text_area(
    "Your message",

    value=st.session_state.user_text,

    placeholder=(
        "Example: I have been feeling lonely and exhausted lately..."
    ),

    height=150,

    label_visibility="collapsed"
)


# Save current text
st.session_state.user_text = user_text


# =========================================================
# CHARACTER COUNT
# =========================================================

st.caption(
    f"{len(user_text)} characters"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "Analyze Text",

    type="primary",

    use_container_width=True,

    key="analyze_button"
)

if st.button("Clear Analysis", key="clear_analysis_button"):
    st.session_state.user_text = ""
    st.session_state.show_result = False
    st.session_state.emotion = ""
    st.session_state.emotion_confidence = 0.0
    st.session_state.emotion_probabilities = {}
    st.session_state.mental_health = ""
    st.session_state.mental_health_confidence = 0.0
    st.rerun()


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not user_text.strip():
        st.session_state.show_result = False
        st.warning("Please enter some text before starting the analysis.")

    elif not models_loaded:
        st.session_state.show_result = False
        st.error("The trained models could not be loaded.")
        st.code(model_error)

    else:
        try:
            result = run_models(user_text)

            st.session_state.show_result = True
            st.session_state.user_text = user_text
            st.session_state.emotion = result["emotion"]
            st.session_state.emotion_confidence = result["emotion_confidence"]
            st.session_state.emotion_probabilities = result["emotion_probabilities"]
            st.session_state.mental_health = result["mental_health"]
            st.session_state.mental_health_confidence = result["mental_health_confidence"]

            # Keep a lightweight session-only history.
            st.session_state.analysis_history.append({
                "Text": user_text[:80] + ("..." if len(user_text) > 80 else ""),
                "Emotion": result["emotion"],
                "Emotion Confidence": f'{result["emotion_confidence"]:.1f}%',
                "Mental Health": result["mental_health"],
                "MH Confidence": f'{result["mental_health_confidence"]:.1f}%'
            })

        except Exception as e:
            st.session_state.show_result = False
            st.error("Analysis could not be completed.")
            st.code(str(e))


# =========================================================
# RESULT
# =========================================================

if st.session_state.show_result:

    st.divider()

    st.markdown(
        "## Analysis Result"
    )

    st.caption(
        "AI-generated linguistic analysis of the submitted text."
    )


    # =====================================================
    # RESULT CARDS
    # =====================================================

    result1, result2 = st.columns(2)


    # =====================================================
    # EMOTION
    # =====================================================

    with result1:

        st.subheader(
            "Emotion"
        )


        st.metric(
            label="Detected Emotion",

            value=st.session_state.emotion
        )


        st.progress(
            min(
                max(
                    st.session_state.emotion_confidence / 100,
                    0.0
                ),
                1.0
            )
        )


        if st.session_state.emotion_confidence > 0:

            st.caption(
                f"Confidence: "
                f"{st.session_state.emotion_confidence:.1f}%"
            )

        else:

            st.caption(
                "Confidence: Not available"
            )


    # =====================================================
    # MENTAL HEALTH
    # =====================================================

    with result2:

        st.subheader(
            "🧠 Mental Health"
        )


        st.metric(
            label="Text Category",

            value=st.session_state.mental_health
        )


        if st.session_state.mental_health_confidence > 0:

            st.progress(
                min(
                    max(
                        st.session_state.mental_health_confidence / 100,
                        0.0
                    ),
                    1.0
                )
            )


            st.caption(
                f"Confidence: "
                f"{st.session_state.mental_health_confidence:.1f}%"
            )

        else:

            st.caption(
                "Prediction: NLP Classification"
            )


    # =====================================================
    # EMOTION PROBABILITY DISTRIBUTION
    # =====================================================

    if st.session_state.emotion_probabilities:
        st.write("")
        st.markdown("### 📊 Emotion Probability Distribution")

        probability_df = pd.DataFrame(
            {
                "Emotion": list(st.session_state.emotion_probabilities.keys()),
                "Probability": list(st.session_state.emotion_probabilities.values())
            }
        ).sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(probability_df["Emotion"], probability_df["Probability"])
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # =====================================================
    # CONFIDENCE / UNCERTAINTY
    # =====================================================

    st.write("")
    st.markdown("### 🎯 Model Confidence")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Emotion Confidence",
            f'{st.session_state.emotion_confidence:.1f}%'
        )
        st.caption(
            confidence_label(st.session_state.emotion_confidence)
        )
        if st.session_state.emotion_confidence < 60:
            st.warning(
                "Low-confidence prediction. Interpret this result cautiously."
            )

    with c2:
        st.metric(
            "Mental Health Confidence",
            f'{st.session_state.mental_health_confidence:.1f}%'
        )
        st.caption(
            confidence_label(st.session_state.mental_health_confidence)
        )
        if st.session_state.mental_health_confidence < 60:
            st.warning(
                "Low-confidence prediction. Interpret this result cautiously."
            )

    # =====================================================
    # TEXT INSIGHTS
    # =====================================================

    st.write("")
    st.markdown("### 📝 Text Insights")

    insights = get_text_insights(st.session_state.user_text)
    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric("Words", insights["Words"])
    with i2:
        st.metric("Sentences", insights["Sentences"])
    with i3:
        st.metric("Characters", insights["Characters"])
    with i4:
        st.metric("Avg. Word Length", insights["Avg. word length"])

    st.caption(
        f'Positive indicators: {insights["Positive indicators"]}  •  '
        f'Negative indicators: {insights["Negative indicators"]}  •  '
        f'Emotional intensity: {insights["Emotional intensity"]}'
    )

    # =====================================================
    # WHY THIS PREDICTION?
    # =====================================================

    st.write("")
    st.markdown("### 🔍 Why This Prediction?")

    emotion_explanation = explain_linear_prediction(
        emotion_model,
        emotion_vectorizer,
        st.session_state.user_text
    )

    mental_explanation = explain_linear_prediction(
        mental_health_model,
        mental_health_vectorizer,
        st.session_state.user_text
    )

    if emotion_explanation:
        st.markdown("**Emotion model — influential TF-IDF features**")
        st.write(
            " • ".join(
                [f"`{word}`" for word, _ in emotion_explanation]
            )
        )
    else:
        st.info(
            "The loaded emotion model does not expose coefficient-based "
            "feature importance, so a word-level explanation is unavailable."
        )

    if mental_explanation:
        st.markdown("**Mental-health model — influential TF-IDF features**")
        st.write(
            " • ".join(
                [f"`{word}`" for word, _ in mental_explanation]
            )
        )
    else:
        st.info(
            "The loaded mental-health model does not expose coefficient-based "
            "feature importance, so a word-level explanation is unavailable."
        )

    st.caption(
        "These are model features associated with the classification, "
        "not clinical indicators or a medical assessment."
    )

    # =====================================================
    # ANALYSIS HISTORY
    # =====================================================

    if st.session_state.analysis_history:
        with st.expander("📜 Session Analysis History"):
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

    # =====================================================
    # COMPARE TWO TEXTS
    # =====================================================

    with st.expander("🆚 Compare Two Texts"):
        st.caption(
            "Compare the linguistic predictions of two texts using the same "
            "trained models."
        )

        compare_a = st.text_area(
            "Text A",
            key="compare_text_a",
            height=120,
            placeholder="Example: I am excited about my new semester."
        )

        compare_b = st.text_area(
            "Text B",
            key="compare_text_b",
            height=120,
            placeholder="Example: I am worried that I will fail my semester."
        )

        if st.button("Compare Texts", key="compare_button"):
            if not compare_a.strip() or not compare_b.strip():
                st.warning("Please enter both Text A and Text B.")
            elif models_loaded:
                try:
                    a = run_models(compare_a)
                    b = run_models(compare_b)

                    compare_df = pd.DataFrame({
                        "Metric": [
                            "Emotion",
                            "Emotion Confidence",
                            "Mental Health Category",
                            "Mental Health Confidence"
                        ],
                        "Text A": [
                            a["emotion"],
                            f'{a["emotion_confidence"]:.1f}%',
                            a["mental_health"],
                            f'{a["mental_health_confidence"]:.1f}%'
                        ],
                        "Text B": [
                            b["emotion"],
                            f'{b["emotion_confidence"]:.1f}%',
                            b["mental_health"],
                            f'{b["mental_health_confidence"]:.1f}%'
                        ]
                    })

                    st.dataframe(
                        compare_df,
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error("Comparison could not be completed.")
                    st.code(str(e))

    st.divider()

    # =====================================================
    # ANALYSIS INFORMATION
    # =====================================================

    st.write("")

    st.markdown(
        "### Analysis Information"
    )


    info1, info2, info3 = st.columns(3)


    with info1:

        st.info(
            "**NLP Analysis**\n\n"
            "Linguistic patterns are analyzed "
            "from the submitted text."
        )


    with info2:

        st.info(
            "**Confidence Score**\n\n"
            "Shows the model's confidence "
            "in its prediction."
        )


    with info3:

        st.info(
            "**Text Based**\n\n"
            "Results are generated from "
            "the submitted text only."
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "**Important Disclaimer**\n\n"
        "MindInsight is an educational NLP-based text "
        "classification system. It does not provide medical "
        "diagnosis, clinical assessment, or professional "
        "medical advice. Predictions represent learned "
        "linguistic patterns from training data and should "
        "not be used as a substitute for qualified "
        "professional support."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "MindInsight • Mental Health & Emotion Analyzer"
)

st.caption(
    "Built with Python • Streamlit • NLP • Machine Learning"
)

st.caption(
    "© 2026 MindInsight Project"
)