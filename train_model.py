import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# ==========================================================
# PHRIS - MULTI-LABEL DISEASE CLASSIFICATION
# ==========================================================

DATA_PATH = "data/Gold_Standard.csv"

MODEL_PATH = "models/disease_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LABEL_PATH = "models/label_binarizer.pkl"


# ==========================================================
# 1. TEXT CLEANING
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol
    text = text.replace("#", " ")

    # Keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# 2. CONVERT ANNOTATION INTO LABEL LIST
# ==========================================================

VALID_LABELS = {
    "AURI",
    "PN",
    "TB",
    "COVID"
}


def convert_labels(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    # X means no disease label
    if value == "" or value.upper() == "X":
        return []

    labels = []

    for label in value.split(","):

        label = label.strip().upper()

        if label in VALID_LABELS:
            labels.append(label)

    return sorted(set(labels))


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("=" * 65)
print("       PHRIS MULTI-LABEL DISEASE CLASSIFICATION")
print("=" * 65)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset not found.")
    print("Expected:")
    print(DATA_PATH)

    exit()

data = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Records:", len(data))
print("Columns:", list(data.columns))


# ==========================================================
# 4. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "post",
    "ag_a",
    "ag_b",
    "lang"
]

for column in required_columns:

    if column not in data.columns:

        print("\nERROR: Missing column:", column)

        print("\nAvailable columns:")

        for col in data.columns:
            print("-", col)

        exit()


# ==========================================================
# 5. REMOVE EMPTY POSTS
# ==========================================================

data = data.dropna(subset=["post"]).copy()

data["post"] = data["post"].astype(str)

data = data[data["post"].str.strip() != ""]

print("\nRecords after removing empty posts:", len(data))


# ==========================================================
# 6. CLEAN TEXT
# ==========================================================

print("\nCleaning text...")

data["clean_text"] = data["post"].apply(clean_text)

data = data[data["clean_text"] != ""]

print("Text cleaning completed.")


# ==========================================================
# 7. CREATE GOLD LABEL
# ==========================================================
#
# The dataset contains two annotations:
#
# ag_a = annotator A
# ag_b = annotator B
#
# We use the agreed annotation when both annotators
# provide the same disease label set.
#
# For disagreement, the union of the disease labels
# is used so that disease signals are not discarded.
#
# ==========================================================

def create_gold_label(row):

    labels_a = convert_labels(row["ag_a"])
    labels_b = convert_labels(row["ag_b"])

    set_a = set(labels_a)
    set_b = set(labels_b)

    # Both annotators agree
    if set_a == set_b:
        return sorted(set_a)

    # If they disagree, combine disease labels
    combined = set_a.union(set_b)

    return sorted(combined)


data["labels"] = data.apply(
    create_gold_label,
    axis=1
)


# ==========================================================
# 8. REMOVE DUPLICATE POSTS
# ==========================================================

data = data.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print("\nRecords after duplicate removal:", len(data))


# ==========================================================
# 9. DISPLAY LABEL INFORMATION
# ==========================================================

print("\nDisease labels used:")

for label in sorted(VALID_LABELS):
    count = data["labels"].apply(
        lambda x: label in x
    ).sum()

    print(label, ":", count)


# ==========================================================
# 10. TRAIN / TEST SPLIT
# ==========================================================

print("\nSplitting dataset...")

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(train_data))
print("Testing records:", len(test_data))


# ==========================================================
# 11. TF-IDF
# ==========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_data["clean_text"]
)

X_test = vectorizer.transform(
    test_data["clean_text"]
)

print("Training TF-IDF shape:", X_train.shape)
print("Testing TF-IDF shape:", X_test.shape)


# ==========================================================
# 12. MULTI-LABEL ENCODING
# ==========================================================

label_binarizer = MultiLabelBinarizer(
    classes=sorted(VALID_LABELS)
)

y_train = label_binarizer.fit_transform(
    train_data["labels"]
)

y_test = label_binarizer.transform(
    test_data["labels"]
)

print("\nClasses:")
print(label_binarizer.classes_)


# ==========================================================
# 13. TRAIN SVM
# ==========================================================

print("\nTraining multi-label SVM...")

model = OneVsRestClassifier(
    LinearSVC(
        C=1.0,
        random_state=42
    )
)

model.fit(
    X_train,
    y_train
)

print("SVM training completed.")


# ==========================================================
# 14. PREDICTION
# ==========================================================

print("\nMaking predictions...")

prediction = model.predict(X_test)


# ==========================================================
# 15. EVALUATION
# ==========================================================

print("\n" + "=" * 65)
print("              MODEL PERFORMANCE")
print("=" * 65)

print(
    classification_report(
        y_test,
        prediction,
        target_names=label_binarizer.classes_,
        zero_division=0
    )
)


# ==========================================================
# 16. SAVE MODEL
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

joblib.dump(
    label_binarizer,
    LABEL_PATH
)


# ==========================================================
# 17. TEST SAMPLE
# ==========================================================

print("\n" + "=" * 65)
print("                 SAMPLE PREDICTION")
print("=" * 65)

sample_text = """
I have fever, cough, difficulty breathing and chest pain.
"""

sample_clean = clean_text(sample_text)

sample_vector = vectorizer.transform(
    [sample_clean]
)

sample_result = model.predict(
    sample_vector
)

predicted_labels = label_binarizer.inverse_transform(
    sample_result
)

print("\nInput:")
print(sample_text)

print("Predicted diseases:")

if predicted_labels and predicted_labels[0]:

    for label in predicted_labels[0]:
        print("-", label)

else:

    print("- X (No relevant disease detected)")


# ==========================================================
# 18. COMPLETED
# ==========================================================

print("\n" + "=" * 65)
print("             PHRIS MODEL COMPLETED")
print("=" * 65)

print("\nSaved files:")

print(MODEL_PATH)
print(VECTORIZER_PATH)
print(LABEL_PATH)