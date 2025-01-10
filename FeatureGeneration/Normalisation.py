from itertools import chain
import math
from collections import Counter
import FileReader

# Input folders for tokenized reviews
INPUT_POS_TOKEN_FOLDER = "coursework/FeatureGeneration/Results/Tokenisation/pos"
INPUT_NEG_TOKEN_FOLDER = "coursework/FeatureGeneration/Results/Tokenisation/neg"
# Output folders for normalized tokens
OUTPUT_POS_TOKEN_FOLDER = "coursework/FeatureGeneration/Results/Normalisation/boosted_tokens/pos"
OUTPUT_NEG_TOKEN_FOLDER = "coursework/FeatureGeneration/Results/Normalisation/boosted_tokens/neg"

# Input folders for n-gram and PoS + noun phrase reviews
INPUT_POS_PHRASE_FOLDER = "coursework/FeatureGeneration/Results/PhraseExtraction/pos"
INPUT_NEG_PHRASE_FOLDER = "coursework/FeatureGeneration/Results/PhraseExtraction/neg"
# Output folders for normalized n-grams and PoS + noun phrases
OUTPUT_POS_PHRASE_FOLDER = "coursework/FeatureGeneration/Results/Normalisation/phrases/pos"
OUTPUT_NEG_PHRASE_FOLDER = "coursework/FeatureGeneration/Results/Normalisation/phrases/neg"

def binary_normalization(data):
    return [
        {term: 1 for term in set(doc)}  # Set ensures each term appears only once per document
        for doc in data
    ]

#######################################################################################################################################

def compute_tf(data):
    return [
        {term: count / sum(term_counts.values()) for term, count in term_counts.items()}
        for term_counts in data
    ]

# Function for computing document frequency (DF)
def compute_df(data):
    df_dict = Counter()
    for term_counts in data:  # Iterate over each document (a dictionary)
        for term in term_counts:  # Iterate over terms in the document
            df_dict[term] += 1
    return df_dict

# Function for computing TF-IDF
def compute_tfidf(data, total_docs):
    # Calculate term frequencies
    tf_data = compute_tf(data)

    # Calculate document frequencies
    df_dict = compute_df(data)

    # Compute TF-IDF
    return [
        {term: tf * math.log(total_docs / (df_dict.get(term, 1))) for term, tf in tf_item.items()}
        for tf_item in tf_data
    ]


#######################################################################################################################################

# Main function to process the dataset and apply normalization
def process_normalization():

    #Read positive and negative reviews for tokenized words
    pos_white_space_tokenized, pos_lemmatized_tokenized, pos_stemmed_tokenized = FileReader.read_boosted_tokenized(INPUT_POS_TOKEN_FOLDER)
    neg_white_space_tokenized, neg_lemmatized_tokenized, neg_stemmed_tokenized = FileReader.read_boosted_tokenized(INPUT_NEG_TOKEN_FOLDER)

    # Read bigrams, trigrams and noun phrases for tokenized words
    pos_bigrams_white_space_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/bigrams_white_space_tokenized.json")
    neg_bigrams_white_space_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/bigrams_white_space_tokenized.json")
    pos_trigrams_white_space_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/trigrams_white_space_tokenized.json")
    neg_trigrams_white_space_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/trigrams_white_space_tokenized.json")
    pos_bigrams_lemmatized_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/bigrams_lemmatized_tokenized.json")
    neg_bigrams_lemmatized_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/bigrams_lemmatized_tokenized.json")
    pos_trigrams_lemmatized_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/trigrams_lemmatized_tokenized.json")
    neg_trigrams_lemmatized_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/trigrams_lemmatized_tokenized.json")
    pos_bigrams_stemmed_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/bigrams_stemmed_tokenized.json")
    neg_bigrams_stemmed_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/bigrams_stemmed_tokenized.json")
    pos_trigrams_stemmed_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/trigrams_stemmed_tokenized.json")
    neg_trigrams_stemmed_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/trigrams_stemmed_tokenized.json")
    pos_noun_phrases_white_space_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/noun_phrases_white_space_tokenized.json")
    neg_noun_phrases_white_space_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/noun_phrases_white_space_tokenized.json")
    pos_noun_phrases_lemmatized_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/noun_phrases_lemmatized_tokenized.json")
    neg_noun_phrases_lemmatized_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/noun_phrases_lemmatized_tokenized.json")
    pos_noun_phrases_stemmed_tokenized = FileReader.load_json(f"{INPUT_POS_PHRASE_FOLDER}/noun_phrases_stemmed_tokenized.json")
    neg_noun_phrases_stemmed_tokenized = FileReader.load_json(f"{INPUT_NEG_PHRASE_FOLDER}/noun_phrases_stemmed_tokenized.json")

    # Compute binary normalization from bigrams, trigrams and noun phrases
    bin_dict_p_bi_w_s_t = binary_normalization(pos_bigrams_white_space_tokenized)
    bin_dict_n_bi_w_s_t = binary_normalization(neg_bigrams_white_space_tokenized)
    bin_dict_p_tri_w_s_t = binary_normalization(pos_trigrams_white_space_tokenized)
    bin_dict_n_tri_w_s_t = binary_normalization(neg_trigrams_white_space_tokenized)
    bin_dict_p_bi_l_t = binary_normalization(pos_bigrams_lemmatized_tokenized)
    bin_dict_n_bi_l_t = binary_normalization(neg_bigrams_lemmatized_tokenized)
    bin_dict_p_tri_l_t = binary_normalization(pos_trigrams_lemmatized_tokenized)
    bin_dict_n_tri_l_t = binary_normalization(neg_trigrams_lemmatized_tokenized)
    bin_dict_p_bi_s_t = binary_normalization(pos_bigrams_stemmed_tokenized)
    bin_dict_n_bi_s_t = binary_normalization(neg_bigrams_stemmed_tokenized)
    bin_dict_p_tri_s_t = binary_normalization(pos_trigrams_stemmed_tokenized)
    bin_dict_n_tri_s_t = binary_normalization(neg_trigrams_stemmed_tokenized)
    bin_dict_p_np_w_s_t = binary_normalization(pos_noun_phrases_white_space_tokenized)
    bin_dict_n_np_w_s_t = binary_normalization(neg_noun_phrases_white_space_tokenized)
    bin_dict_p_np_l_t = binary_normalization(pos_noun_phrases_lemmatized_tokenized)
    bin_dict_n_np_l_t = binary_normalization(neg_noun_phrases_lemmatized_tokenized)
    bin_dict_p_np_s_t = binary_normalization(pos_noun_phrases_stemmed_tokenized)
    bin_dict_n_np_s_t = binary_normalization(neg_noun_phrases_stemmed_tokenized)

    # Compute frequency normalization from the boosted tokenized words
    bin_dict_p_w_s_t = binary_normalization(pos_white_space_tokenized)
    bin_dict_n_w_s_t = binary_normalization(neg_white_space_tokenized)
    bin_dict_p_l_t = binary_normalization(pos_lemmatized_tokenized)
    bin_dict_n_l_t = binary_normalization(neg_lemmatized_tokenized)
    bin_dict_p_s_t = binary_normalization(pos_stemmed_tokenized)
    bin_dict_n_s_t = binary_normalization(neg_stemmed_tokenized)                                             

    # Compute TF-IDF using Scikit-Learn from bigrams, trigrams and noun phrases
    TFIDF_dict_p_bi_w_s_t = compute_tfidf(pos_bigrams_white_space_tokenized, len(pos_bigrams_white_space_tokenized))
    TFIDF_dict_n_bi_w_s_t = compute_tfidf(neg_bigrams_white_space_tokenized, len(neg_bigrams_white_space_tokenized))
    TFIDF_dict_p_tri_w_s_t = compute_tfidf(pos_trigrams_white_space_tokenized, len(pos_trigrams_white_space_tokenized))
    TFIDF_dict_n_tri_w_s_t = compute_tfidf(neg_trigrams_white_space_tokenized, len(neg_trigrams_white_space_tokenized))
    TFIDF_dict_p_bi_l_t = compute_tfidf(pos_bigrams_lemmatized_tokenized, len(pos_bigrams_lemmatized_tokenized))
    TFIDF_dict_n_bi_l_t = compute_tfidf(neg_bigrams_lemmatized_tokenized, len(neg_bigrams_lemmatized_tokenized))
    TFIDF_dict_p_tri_l_t = compute_tfidf(pos_trigrams_lemmatized_tokenized, len(pos_trigrams_lemmatized_tokenized))
    TFIDF_dict_n_tri_l_t = compute_tfidf(neg_trigrams_lemmatized_tokenized, len(neg_trigrams_lemmatized_tokenized))
    TFIDF_dict_p_bi_s_t = compute_tfidf(pos_bigrams_stemmed_tokenized, len(pos_bigrams_stemmed_tokenized))
    TFIDF_dict_n_bi_s_t = compute_tfidf(neg_bigrams_stemmed_tokenized, len(neg_bigrams_stemmed_tokenized))
    TFIDF_dict_p_tri_s_t = compute_tfidf(pos_trigrams_stemmed_tokenized, len(pos_trigrams_stemmed_tokenized))
    TFIDF_dict_n_tri_s_t = compute_tfidf(neg_trigrams_stemmed_tokenized, len(neg_trigrams_stemmed_tokenized))
    TFIDF_dict_p_np_w_s_t = compute_tfidf(pos_noun_phrases_white_space_tokenized, len(pos_noun_phrases_white_space_tokenized))
    TFIDF_dict_n_np_w_s_t = compute_tfidf(neg_noun_phrases_white_space_tokenized, len(neg_noun_phrases_white_space_tokenized))
    TFIDF_dict_p_np_l_t = compute_tfidf(pos_noun_phrases_lemmatized_tokenized, len(pos_noun_phrases_lemmatized_tokenized))
    TFIDF_dict_n_np_l_t = compute_tfidf(neg_noun_phrases_lemmatized_tokenized, len(neg_noun_phrases_lemmatized_tokenized))
    TFIDF_dict_p_np_s_t = compute_tfidf(pos_noun_phrases_stemmed_tokenized, len(pos_noun_phrases_stemmed_tokenized))
    TFIDF_dict_n_np_s_t = compute_tfidf(neg_noun_phrases_stemmed_tokenized, len(neg_noun_phrases_stemmed_tokenized))

    # Compute TF-IDF using Scikit-Learn from the boosted tokenized words
    TFIDF_dict_p_w_s_t = compute_tfidf(pos_white_space_tokenized, len(pos_white_space_tokenized))
    TFIDF_dict_n_w_s_t = compute_tfidf(neg_white_space_tokenized, len(neg_white_space_tokenized))
    TFIDF_dict_p_l_t = compute_tfidf(pos_lemmatized_tokenized, len(pos_lemmatized_tokenized))
    TFIDF_dict_n_l_t = compute_tfidf(neg_lemmatized_tokenized, len(neg_lemmatized_tokenized))
    TFIDF_dict_p_s_t = compute_tfidf(pos_stemmed_tokenized, len(pos_stemmed_tokenized))
    TFIDF_dict_n_s_t = compute_tfidf(neg_stemmed_tokenized, len(neg_stemmed_tokenized))    

    # Save the normalized boosted token vectors to JSON files 
    FileReader.save_to_json(bin_dict_p_w_s_t, f"{OUTPUT_POS_TOKEN_FOLDER}/bin_w_s_t.json")
    FileReader.save_to_json(bin_dict_n_w_s_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/bin_w_s_t.json")
    FileReader.save_to_json(bin_dict_p_l_t, f"{OUTPUT_POS_TOKEN_FOLDER}/bin_l_t.json")
    FileReader.save_to_json(bin_dict_n_l_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/bin_l_t.json")
    FileReader.save_to_json(bin_dict_p_s_t, f"{OUTPUT_POS_TOKEN_FOLDER}/bin_s_t.json")
    FileReader.save_to_json(bin_dict_n_s_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/bin_s_t.json")

    # Save the TF-IDF boosted token vectors to JSON files
    FileReader.save_to_json(TFIDF_dict_p_w_s_t, f"{OUTPUT_POS_TOKEN_FOLDER}/TFIDF_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_w_s_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/TFIDF_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_p_l_t, f"{OUTPUT_POS_TOKEN_FOLDER}/TFIDF_l_t.json")
    FileReader.save_to_json(TFIDF_dict_n_l_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/TFIDF_l_t.json")
    FileReader.save_to_json(TFIDF_dict_p_s_t, f"{OUTPUT_POS_TOKEN_FOLDER}/TFIDF_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_s_t, f"{OUTPUT_NEG_TOKEN_FOLDER}/TFIDF_s_t.json")

    # # Save the normalized phrase vectors to JSON files
    FileReader.save_to_json(bin_dict_p_bi_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_bi_w_s_t.json")
    FileReader.save_to_json(bin_dict_n_bi_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_bi_w_s_t.json")
    FileReader.save_to_json(bin_dict_p_tri_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_tri_w_s_t.json")
    FileReader.save_to_json(bin_dict_n_tri_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_tri_w_s_t.json")
    FileReader.save_to_json(bin_dict_p_bi_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_bi_l_t.json")
    FileReader.save_to_json(bin_dict_n_bi_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_bi_l_t.json")
    FileReader.save_to_json(bin_dict_p_tri_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_tri_l_t.json")
    FileReader.save_to_json(bin_dict_n_tri_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_tri_l_t.json")
    FileReader.save_to_json(bin_dict_p_bi_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_bi_s_t.json")
    FileReader.save_to_json(bin_dict_n_bi_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_bi_s_t.json")
    FileReader.save_to_json(bin_dict_p_tri_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_tri_s_t.json")
    FileReader.save_to_json(bin_dict_n_tri_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_tri_s_t.json")

    FileReader.save_to_json(bin_dict_p_np_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_np_w_s_t.json")
    FileReader.save_to_json(bin_dict_n_np_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_np_w_s_t.json")
    FileReader.save_to_json(bin_dict_p_np_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_np_l_t.json")
    FileReader.save_to_json(bin_dict_n_np_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_np_l_t.json")
    FileReader.save_to_json(bin_dict_p_np_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/bin_np_s_t.json")
    FileReader.save_to_json(bin_dict_n_np_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/bin_np_s_t.json")

    # # Save the TF-IDF phrase vectors to JSON files
    FileReader.save_to_json(TFIDF_dict_p_bi_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_bi_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_bi_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_bi_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_p_tri_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_tri_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_tri_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_tri_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_p_bi_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_bi_l_t.json")
    FileReader.save_to_json(TFIDF_dict_n_bi_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_bi_l_t.json")
    FileReader.save_to_json(TFIDF_dict_p_tri_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_tri_l_t.json")
    FileReader.save_to_json(TFIDF_dict_n_tri_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_tri_l_t.json")
    FileReader.save_to_json(TFIDF_dict_p_bi_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_bi_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_bi_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_bi_s_t.json")
    FileReader.save_to_json(TFIDF_dict_p_tri_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_tri_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_tri_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_tri_s_t.json")

    FileReader.save_to_json(TFIDF_dict_p_np_w_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_np_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_np_w_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_np_w_s_t.json")
    FileReader.save_to_json(TFIDF_dict_p_np_l_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_np_l_t.json")
    FileReader.save_to_json(TFIDF_dict_n_np_l_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_np_l_t.json")
    FileReader.save_to_json(TFIDF_dict_p_np_s_t, f"{OUTPUT_POS_PHRASE_FOLDER}/TFIDF_np_s_t.json")
    FileReader.save_to_json(TFIDF_dict_n_np_s_t, f"{OUTPUT_NEG_PHRASE_FOLDER}/TFIDF_np_s_t.json")
    
    print("Binary normalization and TF-IDF computation completed.")

# Entry point of the script
if __name__ == "__main__":
    process_normalization()