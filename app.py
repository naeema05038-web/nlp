import streamlit as st
import joblib
import numpy as np
import os


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


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not user_text.strip():

        st.session_state.show_result = False

        st.warning(
            "Please enter some text before starting the analysis."
        )

    elif not models_loaded:

        st.session_state.show_result = False

        st.error(
            "The trained models could not be loaded."
        )

        st.code(
            model_error
        )

    else:

        # =================================================
        # MEMBER 2 — EMOTION MODEL
        # =================================================

        emotion_text_vector = (
            emotion_vectorizer.transform(
                [user_text]
            )
        )


        emotion_prediction = (
            emotion_model.predict(
                emotion_text_vector
            )[0]
        )


        emotion = str(
            emotion_prediction
        )


        # =================================================
        # MEMBER 2 — EMOTION CONFIDENCE
        # =================================================

        emotion_confidence = 0.0


        if hasattr(
            emotion_model,
            "predict_proba"
        ):

            try:

                emotion_probabilities = (
                    emotion_model.predict_proba(
                        emotion_text_vector
                    )[0]
                )


                emotion_confidence = (
                    float(
                        np.max(
                            emotion_probabilities
                        )
                    ) * 100
                )

            except Exception:

                emotion_confidence = 0.0


        # =================================================
        # MEMBER 3 — MENTAL HEALTH MODEL
        # =================================================

        mental_health_text_vector = (
            mental_health_vectorizer.transform(
                [user_text]
            )
        )


        mental_health_prediction = (
            mental_health_model.predict(
                mental_health_text_vector
            )[0]
        )


        mental_health = str(
            mental_health_prediction
        )


        # =================================================
        # MEMBER 3 — MENTAL HEALTH CONFIDENCE
        # =================================================

        mental_health_confidence = 0.0


        if hasattr(
            mental_health_model,
            "predict_proba"
        ):

            try:

                mental_health_probabilities = (
                    mental_health_model.predict_proba(
                        mental_health_text_vector
                    )[0]
                )


                mental_health_confidence = (
                    float(
                        np.max(
                            mental_health_probabilities
                        )
                    ) * 100
                )

            except Exception:

                mental_health_confidence = 0.0


        # =================================================
        # SAVE RESULTS
        # =================================================

        st.session_state.show_result = True

        st.session_state.emotion = emotion

        st.session_state.emotion_confidence = (
            emotion_confidence
        )

        st.session_state.mental_health = (
            mental_health
        )

        st.session_state.mental_health_confidence = (
            mental_health_confidence
        )


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