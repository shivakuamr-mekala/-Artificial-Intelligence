# ============================================================
# P22 FAKE NEWS DETECTION — LOGISTIC REGRESSION STREAMLIT APP
# ============================================================

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from scipy.sparse import hstack


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# DEPLOYMENT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODELS_DIR = APP_DIR / "models"
ARTIFACTS_DIR = APP_DIR / "artifacts"
RESULTS_DIR = APP_DIR / "results"

# Raw LIAR dataset location
#
# Expected structure:
#
# P22
# └── deployment
#     ├── app1.py
#     ├── models
#     ├── artifacts
#     ├── results
#     └── data
#         └── raw
#             ├── train.tsv
#             ├── valid.tsv
#             └── test.tsv
#
RAW_DATA_DIR = APP_DIR / "data" / "raw"


# ============================================================
# FILE PATHS
# ============================================================

LOGREG_MODEL_PATH = (
    MODELS_DIR /
    "logistic_regression_tfidf_engineered.pkl"
)

TFIDF_PATH = (
    ARTIFACTS_DIR /
    "tfidf_vectorizer.pkl"
)

LABEL_MAPPING_PATH = (
    ARTIFACTS_DIR /
    "binary_label_mapping.pkl"
)

ENGINEERED_FEATURES_PATH = (
    ARTIFACTS_DIR /
    "logistic_engineered_features.json"
)

LOGREG_RESULTS_PATH = (
    RESULTS_DIR /
    "logistic_regression_results.pkl"
)


# ============================================================
# RAW LIAR DATA PATHS
# ============================================================

TRAIN_RAW_PATH = (
    RAW_DATA_DIR /
    "train.tsv"
)

VALID_RAW_PATH = (
    RAW_DATA_DIR /
    "valid.tsv"
)

TEST_RAW_PATH = (
    RAW_DATA_DIR /
    "test.tsv"
)


# ============================================================
# LABEL DEFINITIONS
# ============================================================

FAKE_LABEL = 0
REAL_LABEL = 1


# ============================================================
# ORIGINAL LIAR SIX-LABEL MAPPING
# ============================================================

LIAR_LABEL_MAPPING = {

    "true": "REAL",

    "mostly-true": "REAL",

    "half-true": "REAL",

    "barely-true": "FAKE",

    "false": "FAKE",

    "pants-fire": "FAKE"
}


# ============================================================
# LOAD DEPLOYMENT ARTIFACTS
# ============================================================

@st.cache_resource
def load_deployment_artifacts():

    # --------------------------------------------------------
    # Load Logistic Regression model
    # --------------------------------------------------------

    with open(
        LOGREG_MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)


    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    with open(
        TFIDF_PATH,
        "rb"
    ) as file:

        tfidf_vectorizer = pickle.load(file)


    # --------------------------------------------------------
    # Load binary label mapping
    # --------------------------------------------------------

    with open(
        LABEL_MAPPING_PATH,
        "rb"
    ) as file:

        label_mapping = pickle.load(file)


    # --------------------------------------------------------
    # Load engineered-feature configuration
    # --------------------------------------------------------

    with open(
        ENGINEERED_FEATURES_PATH,
        "r"
    ) as file:

        engineered_features = json.load(file)


    return (
        model,
        tfidf_vectorizer,
        label_mapping,
        engineered_features
    )


(
    logreg_model,
    tfidf_vectorizer,
    label_mapping,
    engineered_features
) = load_deployment_artifacts()


# ============================================================
# LOAD RAW LIAR DATA
# ============================================================

@st.cache_data
def load_raw_liar_data():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    missing_files = []

    if not TRAIN_RAW_PATH.exists():

        missing_files.append(
            str(TRAIN_RAW_PATH)
        )

    if not VALID_RAW_PATH.exists():

        missing_files.append(
            str(VALID_RAW_PATH)
        )

    if not TEST_RAW_PATH.exists():

        missing_files.append(
            str(TEST_RAW_PATH)
        )


    if missing_files:

        raise FileNotFoundError(
            "The following raw LIAR files were not found:\n\n"
            +
            "\n".join(
                missing_files
            )
        )


    # --------------------------------------------------------
    # Original LIAR dataset columns
    #
    # LIAR TSV files do not contain column names.
    #
    # Column 0 = ID
    # Column 1 = label
    # Column 2 = statement
    # --------------------------------------------------------

    liar_columns = [

        "id",

        "label",

        "statement",

        "subject",

        "speaker",

        "job_title",

        "state_info",

        "party_affiliation",

        "barely_true_counts",

        "false_counts",

        "half_true_counts",

        "mostly_true_counts",

        "pants_fire_counts",

        "context"
    ]


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_df = pd.read_csv(

        TRAIN_RAW_PATH,

        sep="\t",

        header=None,

        names=liar_columns
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_df = pd.read_csv(

        VALID_RAW_PATH,

        sep="\t",

        header=None,

        names=liar_columns
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_df = pd.read_csv(

        TEST_RAW_PATH,

        sep="\t",

        header=None,

        names=liar_columns
    )


    return (

        train_df,

        validation_df,

        test_df
    )


(
    raw_train_df,

    raw_validation_df,

    raw_test_df

) = load_raw_liar_data()


# ============================================================
# PREPARE RAW DATA LOOKUP
# ============================================================

@st.cache_data
def prepare_raw_lookup(

    train_df,

    validation_df,

    test_df
):

    # --------------------------------------------------------
    # Combine all three datasets
    # --------------------------------------------------------

    combined_df = pd.concat(

        [

            train_df,

            validation_df,

            test_df

        ],

        ignore_index=True
    )


    # --------------------------------------------------------
    # Clean statement
    # --------------------------------------------------------

    combined_df["statement"] = (

        combined_df["statement"]

        .fillna("")

        .astype(str)

        .str.strip()
    )


    # --------------------------------------------------------
    # Clean original LIAR label
    # --------------------------------------------------------

    combined_df["label"] = (

        combined_df["label"]

        .fillna("")

        .astype(str)

        .str.strip()

        .str.lower()
    )


    # --------------------------------------------------------
    # Convert six original labels to binary labels
    # --------------------------------------------------------

    combined_df["binary_label"] = (

        combined_df["label"]

        .map(
            LIAR_LABEL_MAPPING
        )
    )


    # --------------------------------------------------------
    # Remove unknown labels
    # --------------------------------------------------------

    combined_df = combined_df[

        combined_df["binary_label"].notna()

    ].copy()


    # --------------------------------------------------------
    # Create normalized statement key
    # --------------------------------------------------------

    combined_df["statement_key"] = (

        combined_df["statement"]

        .str.lower()

        .str.replace(
            r"\s+",
            " ",
            regex=True
        )

        .str.strip()

        .str.strip(
            " \t\n\r.,!?;:\"'()[]{}"
        )
    )


    return combined_df


raw_lookup_df = prepare_raw_lookup(

    raw_train_df,

    raw_validation_df,

    raw_test_df
)


# ============================================================
# NORMALIZE STATEMENT
# ============================================================

def normalize_statement(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = " ".join(
        text.split()
    )

    # Remove common surrounding punctuation
    text = text.strip(
        " \t\n\r.,!?;:\"'()[]{}"
    )

    return text


# ============================================================
# FIND ACTUAL DATASET LABEL
# ============================================================

def find_actual_label(text):

    # --------------------------------------------------------
    # Normalize user input
    # --------------------------------------------------------

    text_key = normalize_statement(
        text
    )


    # --------------------------------------------------------
    # Search normalized statement
    # --------------------------------------------------------

    matches = raw_lookup_df[

        raw_lookup_df["statement_key"]

        == text_key

    ]


    # --------------------------------------------------------
    # No exact match
    # --------------------------------------------------------

    if matches.empty:

        return None


    # --------------------------------------------------------
    # Take first exact match
    # --------------------------------------------------------

    matched_row = matches.iloc[0]


    # --------------------------------------------------------
    # Extract original LIAR label
    # --------------------------------------------------------

    original_label = (

        matched_row["label"]
    )


    # --------------------------------------------------------
    # Extract binary label
    # --------------------------------------------------------

    binary_label = (

        matched_row["binary_label"]
    )


    # --------------------------------------------------------
    # Dataset ID
    # --------------------------------------------------------

    dataset_id = (

        matched_row["id"]
    )


    return {

        "original_label":
            original_label,

        "binary_label":
            binary_label,

        "dataset_id":
            dataset_id
    }


# ============================================================
# ENGINEERED FEATURE CALCULATION
# ============================================================

def calculate_engineered_features(text):

    text = str(text)


    # --------------------------------------------------------
    # Total characters
    # --------------------------------------------------------

    total_characters = max(
        len(text),
        1
    )


    # --------------------------------------------------------
    # Uppercase characters
    # --------------------------------------------------------

    uppercase_characters = sum(

        1

        for character in text

        if character.isupper()
    )


    # --------------------------------------------------------
    # CAPS ratio
    # --------------------------------------------------------

    caps_ratio = (

        uppercase_characters

        /

        total_characters
    )


    # --------------------------------------------------------
    # Punctuation counts
    # --------------------------------------------------------

    exclamation_count = (

        text.count("!")
    )

    question_mark_count = (

        text.count("?")
    )


    # --------------------------------------------------------
    # Return engineered features
    # --------------------------------------------------------

    return np.array(

        [[

            caps_ratio,

            exclamation_count,

            question_mark_count

        ]],

        dtype=float
    )


# ============================================================
# LOGISTIC REGRESSION PREDICTION
# ============================================================

def predict_logistic(text):

    text = str(text)


    # ========================================================
    # STEP 1 — TF-IDF
    # ========================================================

    tfidf_features = (

        tfidf_vectorizer.transform(

            [text]

        )
    )


    # ========================================================
    # STEP 2 — ENGINEERED FEATURES
    # ========================================================

    engineered = (

        calculate_engineered_features(

            text

        )
    )


    # ========================================================
    # STEP 3 — COMBINE FEATURES
    # ========================================================

    combined_features = hstack(

        [

            tfidf_features,

            engineered

        ]
    )


    # ========================================================
    # STEP 4 — PREDICTION
    # ========================================================

    predicted_class = (

        logreg_model.predict(

            combined_features

        )[0]
    )


    # ========================================================
    # STEP 5 — PROBABILITIES
    # ========================================================

    probabilities = (

        logreg_model.predict_proba(

            combined_features

        )[0]
    )


    # Get model's class ordering

    model_classes = list(

        logreg_model.classes_
    )


    fake_index = (

        model_classes.index(

            "FAKE"

        )
    )


    real_index = (

        model_classes.index(

            "REAL"

        )
    )


    fake_probability = float(

        probabilities[

            fake_index

        ]
    )


    real_probability = float(

        probabilities[

            real_index

        ]
    )


    # ========================================================
    # STEP 6 — APPLICATION LABEL
    # ========================================================

    if (

        str(

            predicted_class

        ).upper()

        ==

        "FAKE"

    ):

        predicted_label = FAKE_LABEL

    else:

        predicted_label = REAL_LABEL


    return (

        predicted_label,

        fake_probability,

        real_probability

    )


# ============================================================
# LIME-STYLE WORD IMPORTANCE
# ============================================================

def word_importance(text):

    words = str(text).split()


    if len(words) == 0:

        return pd.DataFrame(

            columns=[

                "Word",

                "Importance"

            ]
        )


    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    (

        original_prediction,

        original_fake_probability,

        original_real_probability

    ) = predict_logistic(text)


    # --------------------------------------------------------
    # Probability of predicted class
    # --------------------------------------------------------

    if (

        original_prediction

        ==

        FAKE_LABEL

    ):

        original_target_probability = (

            original_fake_probability

        )

    else:

        original_target_probability = (

            original_real_probability

        )


    importance_values = []


    # ========================================================
    # REMOVE EACH WORD
    # ========================================================

    for index in range(

        len(words)

    ):

        modified_words = (

            words[:index]

            +

            words[index + 1:]
        )


        modified_text = " ".join(

            modified_words

        )


        if not modified_text.strip():

            importance = 0.0


        else:

            (

                _,

                modified_fake_probability,

                modified_real_probability

            ) = predict_logistic(

                modified_text

            )


            # ------------------------------------------------
            # Always compare against the ORIGINAL predicted
            # class.
            # ------------------------------------------------

            if (

                original_prediction

                ==

                FAKE_LABEL

            ):

                modified_target_probability = (

                    modified_fake_probability

                )

            else:

                modified_target_probability = (

                    modified_real_probability

                )


            # ------------------------------------------------
            # Positive:
            # Removing word decreases predicted probability.
            #
            # Therefore word supports prediction.
            #
            # Negative:
            # Removing word increases predicted probability.
            #
            # Therefore word opposes prediction.
            # ------------------------------------------------

            importance = (

                original_target_probability

                -

                modified_target_probability
            )


        importance_values.append(

            float(importance)

        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    explanation_df = pd.DataFrame(

        {

            "Word":
                words,

            "Importance":
                importance_values

        }
    )


    # --------------------------------------------------------
    # Remove zero importance
    # --------------------------------------------------------

    explanation_df = explanation_df[

        explanation_df["Importance"]

        !=

        0

    ]


    if explanation_df.empty:

        return explanation_df


    # --------------------------------------------------------
    # Sort by absolute importance
    # --------------------------------------------------------

    explanation_df = (

        explanation_df

        .sort_values(

            "Importance",

            key=lambda x: abs(x),

            ascending=False

        )

        .head(15)

        .reset_index(

            drop=True

        )
    )


    return explanation_df


# ============================================================
# HEADER
# ============================================================

st.title(

    "📰 Fake News Detection"

)

st.markdown(

    """
    ### Logistic Regression Fake News Detection

    Enter a news statement and the trained Logistic Regression
    model will classify it as **FAKE** or **REAL**.
    """
)


# ============================================================
# INPUT
# ============================================================

st.header(

    "🔎 Detect Fake News"

)


news_text = st.text_area(

    "Enter a news statement",

    height=180,

    placeholder=(

        "Enter the news statement you want "

        "to classify..."

    )
)


detect_button = st.button(

    "🔍 Detect",

    type="primary"

)


# ============================================================
# DETECTION
# ============================================================

if detect_button:

    if not news_text.strip():

        st.warning(

            "Please enter a news statement."

        )


    else:

        with st.spinner(

            "Analyzing the statement..."

        ):

            try:

                # ====================================================
                # MODEL PREDICTION
                # ====================================================

                (

                    prediction,

                    fake_probability,

                    real_probability

                ) = predict_logistic(

                    news_text

                )


                # ====================================================
                # PREDICTION DISPLAY
                # ====================================================

                st.divider()

                col1, col2 = st.columns(2)


                with col1:

                    st.subheader(

                        "Prediction"

                    )


                    if (

                        prediction

                        ==

                        FAKE_LABEL

                    ):

                        st.error(

                            "🚨 FAKE"

                        )

                    else:

                        st.success(

                            "✅ REAL"

                        )


                with col2:

                    st.subheader(

                        "Model"

                    )

                    st.info(

                        "Logistic Regression"

                    )


                # ====================================================
                # DATASET GROUND TRUTH
                # ====================================================

                actual_result = (

                    find_actual_label(

                        news_text

                    )

                )


                st.divider()

                st.subheader(

                    "📋 Dataset Ground Truth"

                )


                if actual_result is not None:

                    # ------------------------------------------------
                    # Ground-truth information
                    # ------------------------------------------------

                    actual_col1, actual_col2, actual_col3 = (

                        st.columns(3)

                    )


                    with actual_col1:

                        st.metric(

                            "Original LIAR Label",

                            actual_result[

                                "original_label"

                            ]

                        )


                    with actual_col2:

                        st.metric(

                            "Binary Ground Truth",

                            actual_result[

                                "binary_label"

                            ]

                        )


                    with actual_col3:

                        st.metric(

                            "Dataset ID",

                            actual_result[

                                "dataset_id"

                            ]

                        )


                    # ------------------------------------------------
                    # Compare prediction with ground truth
                    # ------------------------------------------------

                    actual_binary_label = (

                        actual_result[

                            "binary_label"

                        ]

                    )


                    prediction_matches = (

                        (

                            prediction

                            ==

                            FAKE_LABEL

                            and

                            actual_binary_label

                            ==

                            "FAKE"

                        )

                        or

                        (

                            prediction

                            ==

                            REAL_LABEL

                            and

                            actual_binary_label

                            ==

                            "REAL"

                        )

                    )


                    if prediction_matches:

                        st.success(

                            "✅ Model prediction matches "

                            "the dataset ground truth."

                        )

                    else:

                        st.error(

                            "❌ Model prediction does not match "

                            "the dataset ground truth."

                        )


                else:

                    st.info(

                        "This statement was not found "

                        "as an exact match in the raw "

                        "LIAR dataset "

                        "(train.tsv, valid.tsv, or test.tsv)."

                    )


                # ====================================================
                # PREDICTED CLASS PROBABILITY
                # ====================================================

                if (

                    prediction

                    ==

                    FAKE_LABEL

                ):

                    probability_value = (

                        fake_probability

                    )

                else:

                    probability_value = (

                        real_probability

                    )


                st.subheader(

                    "Predicted Class Probability"

                )


                st.metric(

                    "Confidence",

                    f"{probability_value * 100:.2f}%"

                )


                st.progress(

                    min(

                        max(

                            probability_value,

                            0.0

                        ),

                        1.0

                    )

                )


                # ====================================================
                # PROBABILITY BREAKDOWN
                # ====================================================

                st.subheader(

                    "Probability Breakdown"

                )


                probability_col1, probability_col2 = (

                    st.columns(2)

                )


                with probability_col1:

                    st.metric(

                        "FAKE Probability",

                        f"{fake_probability * 100:.2f}%"

                    )


                with probability_col2:

                    st.metric(

                        "REAL Probability",

                        f"{real_probability * 100:.2f}%"

                    )


                # ====================================================
                # ENGINEERED FEATURES
                # ====================================================

                st.divider()

                st.subheader(

                    "Engineered Features"

                )


                engineered_values = (

                    calculate_engineered_features(

                        news_text

                    )[0]

                )


                feature_df = pd.DataFrame(

                    {

                        "Feature": [

                            "caps_ratio",

                            "exclamation_count",

                            "question_mark_count"

                        ],

                        "Value": [

                            engineered_values[0],

                            engineered_values[1],

                            engineered_values[2]

                        ]

                    }

                )


                st.dataframe(

                    feature_df,

                    use_container_width=True,

                    hide_index=True

                )


                # ====================================================
                # WORD IMPORTANCE
                # ====================================================

                st.divider()

                st.subheader(

                    "🧠 Word Importance"

                )


                st.caption(

                    "LIME-style perturbation explanation: "

                    "each word is removed individually and "

                    "the change in the probability of the "

                    "predicted class is measured."

                )


                explanation_df = (

                    word_importance(

                        news_text

                    )

                )


                if len(

                    explanation_df

                ) > 0:

                    st.bar_chart(

                        explanation_df.set_index(

                            "Word"

                        )[

                            "Importance"

                        ]

                    )


                    st.dataframe(

                        explanation_df,

                        use_container_width=True,

                        hide_index=True

                    )

                else:

                    st.info(

                        "No measurable word-level "

                        "importance was found."

                    )


            except Exception as error:

                st.error(

                    "Prediction failed."

                )

                st.exception(

                    error

                )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header(

    "⚙️ How This Model Works"

)

st.markdown(

    """
    ### Logistic Regression Pipeline

    **News Statement**

    ↓

    **TF-IDF Vectorization**

    ↓

    **10,000 TF-IDF Features**

    +

    **3 Engineered Features**

    - `caps_ratio`
    - `exclamation_count`
    - `question_mark_count`

    ↓

    **10,003 Total Features**

    ↓

    **Trained Logistic Regression**

    ↓

    **FAKE / REAL**

    ↓

    **Prediction Probability**

    ↓

    **Raw LIAR Dataset Ground Truth**

    - Original six-class LIAR label
    - Binary FAKE / REAL label
    - Dataset ID
    """
)


st.caption(

    "P22 — Fake News Detection | "

    "Logistic Regression Deployment"

)