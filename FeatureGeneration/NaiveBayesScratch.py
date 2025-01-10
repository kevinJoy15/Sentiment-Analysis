import os
from math import log
from collections import Counter
import FileReader
import json

def train_naive_bayes_from_scratch(pos_data, neg_data):
    vocab = set()
    pos_counts = {}
    neg_counts = {}
    total_pos_terms = 0
    total_neg_terms = 0

    # Aggregate term frequencies
    for review in pos_data:
        for term, count in review.items():
            vocab.add(term)
            pos_counts[term] = pos_counts.get(term, 0) + count
            total_pos_terms += count

    for review in neg_data:
        for term, count in review.items():
            vocab.add(term)
            neg_counts[term] = neg_counts.get(term, 0) + count
            total_neg_terms += count

    # Compute priors for positive and negative classes
    prior_pos = len(pos_data) / (len(pos_data) + len(neg_data)) 
    prior_neg = len(neg_data) / (len(pos_data) + len(neg_data))

    # Compute likelihoods with Laplace smoothing, in this case I have decided to use a smoothing factor of 1.0
    smoothing_factor = 1.0
    likelihoods_pos = {term: (pos_counts.get(term, 0) + smoothing_factor) /
                              (total_pos_terms + smoothing_factor * len(vocab))
                       for term in vocab}
    likelihoods_neg = {term: (neg_counts.get(term, 0) + smoothing_factor) /
                              (total_neg_terms + smoothing_factor * len(vocab))
                       for term in vocab}

    return prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab

def predict_naive_bayes(review, prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab):
    """Predict class using term frequencies."""
    # Taking the log of the probabilities to avoid underflow
    log_prob_pos = log(prior_pos)
    log_prob_neg = log(prior_neg)

    # Compute probabilities for terms in the review
    for term, count in review.items():
        if term in vocab:
            log_prob_pos += count * log(likelihoods_pos.get(term, 1 / len(vocab)))
            log_prob_neg += count * log(likelihoods_neg.get(term, 1 / len(vocab)))
    
    # Predict the REVIEW with the highest probability
    return 1 if log_prob_pos > log_prob_neg else 0

def evaluate_naive_bayes(test_reviews, y_test, prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab):
    """Evaluate the Naive Bayes classifier."""
    predictions = [predict_naive_bayes(review, prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab) for review in test_reviews]
    accuracy = sum(p == y for p, y in zip(predictions, y_test)) / len(y_test)
    return accuracy

def main():
    # File paths
    chosen_normalisation = "TFIDF_bi_w_s_t.json"
    
    pos_file = f"coursework/FeatureGeneration/Results/Normalisation/phrases/pos/{chosen_normalisation}"
    neg_file = f"coursework/FeatureGeneration/Results/Normalisation/phrases/neg/{chosen_normalisation}"
    test_folder = "coursework/dataset/evaluationDataset"

    # Load data
    pos_data = FileReader.load_json(pos_file)
    neg_data = FileReader.load_json(neg_file)
    test_reviews, y_test = load_test_reviews(test_folder)

    # Train Naive Bayes
    prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab = train_naive_bayes_from_scratch(pos_data, neg_data)

    # Evaluate on test data
    accuracy = evaluate_naive_bayes(test_reviews, y_test, prior_pos, prior_neg, likelihoods_pos, likelihoods_neg, vocab)

    file_path = 'coursework/FeatureGeneration/Results/Phrase_Result.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract the accuracy and f1_score for "TFIDF_bi_w_s_t.json"
    result = data[0].get(chosen_normalisation, {})
    print(f"MultinomialNB: {result}")
    print(f"Accuracy: {accuracy:.2f}")

def load_test_reviews(test_folder):
    """Load test reviews and labels based on file prefixes, and convert reviews to term frequency dictionaries."""
    test_data = []
    y_test = []
    for filename in os.listdir(test_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(test_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review = file.read().strip()
                # Convert review to term frequency dictionary
                term_frequency = Counter(review.split())
                test_data.append(term_frequency)
                y_test.append(1 if filename.startswith("pos_") else 0)
    return test_data, y_test

if __name__ == "__main__":
    main()
