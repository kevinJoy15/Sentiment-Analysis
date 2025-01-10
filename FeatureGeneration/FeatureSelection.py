import os
import FileReader
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score


# We use this to check for phrase (n-gram and PoS phrase) normalisation
# PATH_NORMALIZATION = "coursework/FeatureGeneration/Results/Normalisation/phrases/pos"
# NORMALISED = "coursework/FeatureGeneration/Results/Normalisation/phrases"

# We use this to check for boosted token normalisation
PATH_NORMALIZATION = "coursework/FeatureGeneration/Results/Normalisation/boosted_tokens/pos"
NORMALISED = "coursework/FeatureGeneration/Results/Normalisation/boosted_tokens"

# This is used to check on both eval and test datasets
PATH_TEST = "coursework/dataset/evaluationDataset"
# PATH_TEST = "coursework/dataset/testDataset"

def prepare_features(pos_data, neg_data):
    """Prepare features and labels for training/testing."""
    vectorizer = DictVectorizer(sparse=True)
    X = pos_data + neg_data
    y = [1] * len(pos_data) + [0] * len(neg_data)  # 1 for positive, 0 for negative
    return vectorizer.fit_transform(X), y, vectorizer

def train_naive_bayes(X, y, model_type):
    """Train a Naive Bayes model (Multinomial or Bernoulli)."""
    if model_type == 'multinomial':
        model = MultinomialNB()
    elif model_type == 'bernoulli':
        model = BernoulliNB()

    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    f1_score = report["weighted avg"]["f1-score"]           # This is used to get the f1 score
    return {"accuracy": round(accuracy, 3), "f1_score": round(f1_score, 3)}         # returns a dictionary with accuracy and f1 score

def load_test_reviews(test_folder):
    """Load test reviews and labels based on file prefixes."""
    test_data = []
    y_test = []
    for filename in os.listdir(test_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(test_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review = file.read().strip()
                test_data.append(review)
                y_test.append(1 if filename.startswith("pos_") else 0)
    return test_data, y_test

def main():
    # Define normalization types and corresponding file paths
    TFIDF_results = {}
    BINARY_results = {}
    comb_dic = []
    OUTPUT = "coursework/FeatureGeneration/Results/Result.json"

    test_reviews, y_test = load_test_reviews(PATH_TEST)
    # Prepare test data
    
    for files in os.listdir(PATH_NORMALIZATION):
        
        # Define file paths for positive and negative data
        pos_file = f"{NORMALISED}/pos/{files}"
        neg_file = f"{NORMALISED}/neg/{files}"
        
        # Load data
        pos_data = FileReader.load_json(pos_file)
        neg_data = FileReader.load_json(neg_file)
        
        # Prepare features and labels
        X, y, vectorizer = prepare_features(pos_data, neg_data)

        test_data_dicts = [{word: 1 for word in review.split()} for review in test_reviews]
        X_test = vectorizer.transform(test_data_dicts)
        
        # Train models
        if files.startswith("bin"):
            bernoulli_model = train_naive_bayes(X, y, model_type="bernoulli")
            bernoulli_results = evaluate_model(bernoulli_model, X_test, y_test)
            BINARY_results[f"{files}"] = bernoulli_results

        elif files.startswith("TFIDF"):
            multinomial_model = train_naive_bayes(X, y, model_type="multinomial")
            multinomial_results = evaluate_model(multinomial_model, X_test, y_test)
            TFIDF_results[f"{files}"] = multinomial_results

    comb_dic = [TFIDF_results, BINARY_results]
    FileReader.save_to_json(comb_dic, OUTPUT)
    
    print("Predictions have been saved")

if __name__ == "__main__":
    main()
