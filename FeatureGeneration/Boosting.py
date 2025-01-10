from nltk.corpus import wordnet as wn
import FileReader
from collections import Counter

def boost_features(tokens_list, BOOST_FACTOR):
    """
    Boost token frequencies based on semantic relations from WordNet.
    """
    boosted_tokens_list = []
    
    for tokens in tokens_list:
        token_freq = Counter(tokens)  # Calculate token frequencies
        boosted_freq = token_freq.copy()
        
        for token in token_freq:
            synsets = wn.synsets(token)  # Get WordNet synsets for the token
            if synsets:
                # Boost frequency if related semantic concepts are found
                boosted_freq[token] *= BOOST_FACTOR

        # Create a boosted list of tokens
        boosted_tokens = []
        for token, freq in boosted_freq.items():
            boosted_tokens.extend([token] * int(freq))  # Add boosted frequency

        boosted_tokens_list.append(boosted_tokens)

    return boosted_tokens_list

def process_with_boosting(BOOST_FACTOR):
    """
    Process the dataset and apply boosting to features.
    """
    # Load tokenized reviews
    pos_w_s_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/pos/whitespace_filtered.json")
    neg_w_s_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/neg/whitespace_filtered.json")
    pos_s_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/pos/stemmed_filtered.json")
    neg_s_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/neg/stemmed_filtered.json")
    pos_l_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/pos/lemmatized_filtered.json")
    neg_l_t = FileReader.load_json("coursework/FeatureGeneration/Results/Tokenisation/neg/lemmatized_filtered.json")
    
    # Boost features
    boosted_pos_w_s_t = boost_features(pos_w_s_t, BOOST_FACTOR)
    boosted_neg_w_s_t = boost_features(neg_w_s_t, BOOST_FACTOR)
    boosted_pos_s_t = boost_features(pos_s_t, BOOST_FACTOR)
    boosted_neg_s_t = boost_features(neg_s_t, BOOST_FACTOR)
    boosted_pos_l_t = boost_features(pos_l_t, BOOST_FACTOR)
    boosted_neg_l_t = boost_features(neg_l_t, BOOST_FACTOR)
    
    # Save boosted tokens
    FileReader.save_to_json([Counter(val) for val in boosted_pos_w_s_t], "coursework/FeatureGeneration/Results/Tokenisation/pos/boosted_w_s_t.json")
    FileReader.save_to_json([Counter(val) for val in boosted_neg_w_s_t], "coursework/FeatureGeneration/Results/Tokenisation/neg/boosted_w_s_t.json")
    FileReader.save_to_json([Counter(val) for val in boosted_pos_s_t], "coursework/FeatureGeneration/Results/Tokenisation/pos/boosted_s_t.json")
    FileReader.save_to_json([Counter(val) for val in boosted_neg_s_t], "coursework/FeatureGeneration/Results/Tokenisation/neg/boosted_s_t.json")
    FileReader.save_to_json([Counter(val) for val in boosted_pos_l_t], "coursework/FeatureGeneration/Results/Tokenisation/pos/boosted_l_t.json")
    FileReader.save_to_json([Counter(val) for val in boosted_neg_l_t], "coursework/FeatureGeneration/Results/Tokenisation/neg/boosted_l_t.json")
    
    print("Feature boosting completed and saved.")

if __name__ == "__main__":
    process_with_boosting(BOOST_FACTOR = 1.5)
