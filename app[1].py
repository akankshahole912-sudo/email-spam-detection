import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="wide")

st.title("📧 Email Spam Detection Using Machine Learning")
st.caption("A beginner-friendly ML project using TF-IDF and Naive Bayes / Logistic Regression.")

DEFAULT_DATA = Path(__file__).parent / "data" / "spam.csv"

@st.cache_data
def load_dataset(path, uploaded_bytes=None):
    if uploaded_bytes is not None:
        import io
        raw = pd.read_csv(io.BytesIO(uploaded_bytes), encoding="utf-8", on_bad_lines="skip")
    else:
        raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    # Support common dataset column names (e.g. UCI SMS Spam Collection CSV)
    lower_to_original = {str(c).strip().lower(): c for c in raw.columns}
    label_col = next((lower_to_original[k] for k in ["label", "category", "v1", "class", "type"] if k in lower_to_original), None)
    text_col = next((lower_to_original[k] for k in ["message", "text", "email", "email_text", "v2", "content"] if k in lower_to_original), None)
    if label_col is None or text_col is None:
        # Handle headerless UCI SMS dataset: first two columns are label and text
        if raw.shape[1] >= 2:
            raw = raw.rename(columns={raw.columns[0]: "label", raw.columns[1]: "message"})
            label_col, text_col = "label", "message"
        else:
            raise ValueError("Could not identify label/text columns. Please upload a CSV with columns like label,message or v1,v2.")
    data = raw[[label_col, text_col]].copy()
    data.columns = ["label", "message"]
    data = data.dropna()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    label_map = {
        "spam": "spam", "ham": "ham", "not spam": "ham", "legitimate": "ham",
        "1": "spam", "0": "ham", "junk": "spam"
    }
    data["label"] = data["label"].map(lambda x: label_map.get(x, x))
    data = data[data["label"].isin(["spam", "ham"])]
    data = data[data["message"].str.len() > 0].drop_duplicates()
    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain both spam and ham/not-spam examples.")
    return data

@st.cache_resource
def train_model(texts, labels, algorithm):
    if algorithm == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = MultinomialNB()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
        ("classifier", clf)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "F1-score": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "y_test": y_test,
        "predictions": predictions
    }
    return model, metrics

with st.sidebar:
    st.header("Project settings")
    algorithm = st.selectbox("Choose ML algorithm", ["Naive Bayes", "Logistic Regression"])
    st.write("Upload your own labelled CSV if you have one.")
    uploaded = st.file_uploader("Dataset CSV", type=["csv"])
    st.info("Expected columns: label,message (also supports UCI v1,v2). Labels should be spam and ham.")

try:
    if uploaded:
        data = load_dataset(None, uploaded.getvalue())
        dataset_name = uploaded.name
    elif DEFAULT_DATA.exists():
        data = load_dataset(DEFAULT_DATA)
        dataset_name = "data/spam.csv"
    else:
        st.error("Dataset not found. Put a labelled CSV at data/spam.csv or upload one in the sidebar.")
        st.stop()

    if len(data) < 10 or data["label"].value_counts().min() < 2:
        st.error("Dataset is too small for a reliable train/test split. Please use a larger labelled dataset.")
        st.stop()

    model, metrics = train_model(tuple(data["message"]), tuple(data["label"]), algorithm)

    tab1, tab2, tab3 = st.tabs(["🔎 Test a Message", "📊 Model Performance", "📁 Dataset Overview"])

    with tab1:
        st.subheader("Check whether a message is spam")
        message = st.text_area("Paste an email or SMS message here", height=150,
            placeholder="Example: Congratulations! You won a prize. Click here to claim...")
        if st.button("Detect Spam", type="primary"):
            if not message.strip():
                st.warning("Please enter a message first.")
            else:
                result = model.predict([message])[0]
                if hasattr(model.named_steps["classifier"], "predict_proba"):
                    probs = model.predict_proba([message])[0]
                    classes = list(model.named_steps["classifier"].classes_)
                    spam_probability = float(probs[classes.index("spam")]) if "spam" in classes else 0.0
                else:
                    spam_probability = None
                if result == "spam":
                    st.error("🚨 Prediction: SPAM")
                    st.write("This message resembles examples labelled as spam in the training dataset.")
                else:
                    st.success("✅ Prediction: NOT SPAM")
                    st.write("This message resembles examples labelled as legitimate in the training dataset.")
                if spam_probability is not None:
                    st.metric("Estimated spam probability", f"{spam_probability:.1%}")
                st.caption("This is a machine-learning prediction, not a guarantee that a message is safe.")

    with tab2:
        st.subheader(f"Evaluation — {algorithm}")
        cols = st.columns(4)
        for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"]):
            col.metric(key, f"{metrics[key]:.1%}")
        st.caption("Metrics are calculated on a held-out 20% test split. Results depend on the dataset.")
        cm = confusion_matrix(metrics["y_test"], metrics["predictions"], labels=["ham", "spam"])
        st.write("Confusion matrix (rows = actual, columns = predicted)")
        st.dataframe(pd.DataFrame(cm, index=["Actual NOT SPAM", "Actual SPAM"],
                                  columns=["Predicted NOT SPAM", "Predicted SPAM"]), use_container_width=True)
        report = classification_report(metrics["y_test"], metrics["predictions"],
                                       labels=["ham", "spam"], target_names=["NOT SPAM", "SPAM"],
                                       zero_division=0, output_dict=True)
        st.write("Classification report")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    with tab3:
        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total messages", f"{len(data):,}")
        c2.metric("Spam messages", f"{(data['label'] == 'spam').sum():,}")
        c3.metric("Not-spam messages", f"{(data['label'] == 'ham').sum():,}")
        st.bar_chart(data["label"].value_counts().rename(index={"ham": "NOT SPAM", "spam": "SPAM"}))
        st.write("Sample rows")
        st.dataframe(data.head(20), use_container_width=True)
        st.download_button("Download cleaned dataset", data.to_csv(index=False).encode("utf-8"),
                           file_name="cleaned_spam_dataset.csv", mime="text/csv")
except Exception as e:
    st.error(f"Could not load/train the dataset: {e}")
    st.markdown("Check that your CSV has a message column and a label column, with both spam and ham examples.")
