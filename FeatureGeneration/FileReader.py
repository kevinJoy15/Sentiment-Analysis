import json
import os

def read_tokenized(folder_path):

    white_space_tokenized = []
    lemmatized_tokenized = []
    stemmed_tokenized = []

    for filename in os.listdir(folder_path):
        if filename == "whitespace_filtered.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                white_space_tokenized = json.load(file)
        elif filename == "lemmatized_filtered.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lemmatized_tokenized = json.load(file)
        elif filename == "stemmed_filtered.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                stemmed_tokenized = json.load(file)

    return white_space_tokenized, lemmatized_tokenized, stemmed_tokenized

def read_boosted_tokenized(folder_path):

    white_space_tokenized = []
    lemmatized_tokenized = []
    stemmed_tokenized = []

    for filename in os.listdir(folder_path):
        if filename == "boosted_w_s_t.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                white_space_tokenized = json.load(file)

        elif filename == "boosted_l_t.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lemmatized_tokenized = json.load(file)

        elif filename == "boosted_s_t.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                stemmed_tokenized = json.load(file)

    return white_space_tokenized, lemmatized_tokenized, stemmed_tokenized

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to read reviews from a folder
def read_reviews(folder_path):
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review = file.read().strip()
                reviews.append(review)
    return reviews

def clear_folder(directory_path):
    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def load_files(folder, label):
    texts, labels = [], []
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), "r", encoding="utf-8") as file:
            texts.append(file.read().strip())
            if label is None:
                if file_name.startswith("neg_"):
                    label = 0
                else:
                    label = 1
            labels.append(label)
    return texts, labels