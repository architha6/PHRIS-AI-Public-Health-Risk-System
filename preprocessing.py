import re
import os
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_data(data, text_column="post"):

    if text_column not in data.columns:
        raise ValueError(
            f"Column '{text_column}' was not found."
        )

    data = data.dropna(subset=[text_column]).copy()

    data["clean_text"] = data[text_column].apply(clean_text)

    data = data[data["clean_text"] != ""]

    data = data.drop_duplicates(
        subset=["clean_text"]
    )

    data = data.reset_index(drop=True)

    return data


def load_dataset(file_path):

    if not os.path.exists(file_path):
        print("ERROR: Dataset not found.")
        print("File:", file_path)
        return None

    try:
        data = pd.read_csv(file_path)

        print("\nDataset loaded successfully.")
        print("Number of records:", len(data))
        print("Columns:", list(data.columns))

        return data

    except Exception as error:

        print("ERROR while loading dataset:")
        print(error)

        return None


def main():

    print("=" * 50)
    print("       PHRIS TEXT PREPROCESSING")
    print("=" * 50)

    dataset_path = "data/Gold_Standard.csv"
    output_path = "data/processed_dataset.csv"

    data = load_dataset(dataset_path)

    if data is None:
        return

    if "post" not in data.columns:

        print("\nERROR:")
        print("The dataset does not contain a 'post' column.")

        print("\nAvailable columns:")

        for column in data.columns:
            print("-", column)

        return

    print("\nPreprocessing data...")

    processed_data = preprocess_data(
        data,
        text_column="post"
    )

    print(
        "Records after preprocessing:",
        len(processed_data)
    )

    try:

        processed_data.to_csv(
            output_path,
            index=False
        )

        print("\nProcessed dataset saved to:")
        print(output_path)

    except Exception as error:

        print("\nERROR while saving dataset:")
        print(error)

        return

    print("\n" + "=" * 50)
    print("CLEANING EXAMPLES")
    print("=" * 50)

    for i in range(min(5, len(processed_data))):

        print("\nOriginal:")
        print(processed_data.iloc[i]["post"])

        print("Cleaned:")
        print(processed_data.iloc[i]["clean_text"])

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()