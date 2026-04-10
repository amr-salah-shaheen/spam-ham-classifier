import joblib
import streamlit as st

st.set_page_config(page_title="Spam/Ham Classifier", page_icon="📩", layout="wide")

MODEL_PATH = "model/best_spam_model.pkl"

@st.cache_resource
def load_pipeline(model_path: str):
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        for key in ("model", "pipeline", "best_model"):
            if key in loaded:
                return loaded[key]
    return loaded

def to_spam_ham(raw_label) -> str:
    value = str(raw_label).strip().lower()
    if value in {"1", "spam"}:
        return "spam"
    if value in {"0", "ham"}:
        return "ham"
    return str(raw_label)

def predict_texts(pipeline, texts: list[str]):
    return pipeline.predict(texts)

st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
    }
    div[data-testid="stTextArea"] textarea {
        font-size: 24px !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        font-size: 20px !important;
    }
    div[data-testid="stButton"] > button,
    div.stButton > button,
    button[kind="primary"] {
        display: block;
        margin: 0 auto;
        width: 130px;
        height: 58px;
    }
    div[data-testid="stButton"] > button p,
    div.stButton > button p,
    button[kind="primary"] p {
        font-size: 28px !important;
        font-weight: 300 !important;
    }
    .prediction-text {
        text-align: center;
        font-size: 42px;
        font-weight: 600;
        margin-top: 12px;
        color: white;
        padding: 14px 18px;
        border-radius: 12px;
    }
    .prediction-ham {
        background-color: #16a34a;
    }
    .prediction-spam {
        background-color: #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="center-text">📩 Spam/Ham Text Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; font-size:26px;">Classify SMS/email messages as spam or ham</p>',
    unsafe_allow_html=True,
)

try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model. Details: `{exc}`")
    st.stop()

left, center, right = st.columns([1, 2, 1])
with center:
    st.markdown(
        '<p class="center-text" style="font-size:24px;"><strong>Enter message</strong></p>',
        unsafe_allow_html=True,
    )
    user_text = st.text_area(
        "Enter message",
        height=200,
        placeholder="Type SMS/email text...",
        label_visibility="collapsed",
    )
    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                preds = predict_texts(pipeline=pipeline, texts=[user_text])
                label = to_spam_ham(preds[0])
                color_class = "prediction-ham" if label.lower() == "ham" else "prediction-spam"
                st.markdown(
                    f'<p class="prediction-text {color_class}">Prediction: {label.upper()}</p>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Prediction failed: `{exc}`")
