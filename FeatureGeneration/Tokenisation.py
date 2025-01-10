from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import FileReader
import re


# Initialize NLTK components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

POSITIVE_FOLDER = "coursework/dataset/trainDataset/pos"
NEGATIVE_FOLDER = "coursework/dataset/trainDataset/neg"

IGNORE_PATTERN = re.compile(r'[^\w\s]|<\s*[^>\s/]+\s*/?\s*>')

# Tokenization functions
def whitespace_tokenizer(text):
    tokens = text.split()
    return [token for token in tokens if token.lower() and (not IGNORE_PATTERN.fullmatch(token)) and token.lower() not in stop_words]

def lemmatize_tokenizer(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() and not IGNORE_PATTERN.fullmatch(token) and token.lower() not in stop_words]
    return lemmatized_tokens

def stem_tokenizer(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens if token.lower() and not IGNORE_PATTERN.fullmatch(token) and token.lower() not in stop_words]
    return stemmed_tokens

# Function to filter tokens based on frequency
def filter_low_frequency_tokens(tokens_list, frequency_threshold):

    # Filter tokens based on frequency threshold
    filtered_tokens_list = []
    for tokens in tokens_list:

        token_freq = Counter(tokens)
    
        filtered_tokens = [token for token in tokens if token_freq[token] < frequency_threshold]
        filtered_tokens_list.append(filtered_tokens)

    return filtered_tokens_list

def doesTokenExist(reviews, frequency_threshold, folder_path):
    whitespace_tokens = [whitespace_tokenizer(review) for review in reviews]
    lemmatized_tokens = [lemmatize_tokenizer(review) for review in reviews]
    stemmed_tokens = [stem_tokenizer(review) for review in reviews]

    whitespace_filtered = filter_low_frequency_tokens(whitespace_tokens, frequency_threshold)
    lemmatized_filtered = filter_low_frequency_tokens(lemmatized_tokens, frequency_threshold)
    stemmed_filtered = filter_low_frequency_tokens(stemmed_tokens, frequency_threshold)

    FileReader.save_to_json(whitespace_filtered, f"{folder_path}whitespace_filtered.json")
    FileReader.save_to_json(lemmatized_filtered, f"{folder_path}lemmatized_filtered.json")
    FileReader.save_to_json(stemmed_filtered, f"{folder_path}stemmed_filtered.json")
    
# Main function to process the dataset
def process_dataset(frequency_threshold):
    # Read positive and negative reviews
    pos_reviews = FileReader.read_reviews(POSITIVE_FOLDER)
    neg_reviews = FileReader.read_reviews(NEGATIVE_FOLDER)

    doesTokenExist(pos_reviews, frequency_threshold, "coursework/FeatureGeneration/Results/Tokenisation/pos/")
    doesTokenExist(neg_reviews, frequency_threshold, "coursework/FeatureGeneration/Results/Tokenisation/neg/")

    print(f"Token filtering completed with frequency threshold = {frequency_threshold}.")
    print("completed tokenisation")

# Entry point of the script
if __name__ == "__main__":
    # You can experiment with different frequency thresholds here
    process_dataset(frequency_threshold=5)
