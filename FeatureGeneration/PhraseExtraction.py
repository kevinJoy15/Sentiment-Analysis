from nltk import pos_tag
from collections import Counter
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
import FileReader

INPUT_FOLDER = "coursework/FeatureGeneration/Results/Tokenisation"

POS_OUTPUT_FOLDER = "coursework/FeatureGeneration/Results/PhraseExtraction/pos"
NEG_OUTPUT_FOLDER = "coursework/FeatureGeneration/Results/PhraseExtraction/neg"

def combine(n_gram, token):
    token_dict = [dict(Counter(lst)) for lst in token]
    return [{**d1, **d2} for d1, d2 in zip(n_gram, token_dict)]

# Function to extract bigrams using NLTK's BigramCollocationFinder
def extract_bigrams(tokens):
    finder = BigramCollocationFinder.from_words(tokens)
    bigrams = {" ".join(bigram): freq for bigram, freq in finder.ngram_fd.items()}
    return bigrams

# Extract trigrams using NLTK's TrigramCollocationFinder
def extract_trigrams(tokens):
    finder = TrigramCollocationFinder.from_words(tokens)
    trigrams = {" ".join(trigram): freq for trigram, freq in finder.ngram_fd.items()}
    return trigrams

# Extract noun phrases using PoS tagging
def extract_noun_phrases(tokens):
    pos_tags = pos_tag(tokens)
    noun_phrases = []
    current_phrase = []
    for word, tag in pos_tags:
        if tag.startswith("NN"):  # Noun tags
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(" ".join(current_phrase))
            current_phrase = []
    if current_phrase:
        noun_phrases.append(" ".join(current_phrase))
    return Counter(noun_phrases)

# Main function to process the dataset and extract phrases
def process_phrases():

    pos_white_space_tokenized, pos_lemmatized_tokenized, pos_stemmed_tokenized = FileReader.read_tokenized(f"{INPUT_FOLDER}/pos")
    neg_white_space_tokenized, neg_lemmatized_tokenized, neg_stemmed_tokenized = FileReader.read_tokenized(f"{INPUT_FOLDER}/neg")


    # Extract N-grams (bigrams and trigrams)
    pos_bigrams_white_space_tokenized = [extract_bigrams(review) for review in pos_white_space_tokenized]
    pos_trigrams_white_space_tokenized = [extract_trigrams(review) for review in pos_white_space_tokenized]
    pos_bigrams_lemmatized_tokenized = [extract_bigrams(review) for review in pos_lemmatized_tokenized]
    pos_trigrams_lemmatized_tokenized = [extract_trigrams(review) for review in pos_lemmatized_tokenized]
    pos_bigrams_stemmed_tokenized = [extract_bigrams(review) for review in pos_stemmed_tokenized]
    pos_trigrams_stemmed_tokenized = [extract_trigrams(review) for review in pos_stemmed_tokenized]


    neg_bigrams_white_space_tokenized = [extract_bigrams(review) for review in neg_white_space_tokenized]
    neg_trigrams_white_space_tokenized = [extract_trigrams(review) for review in neg_white_space_tokenized]
    neg_bigrams_lemmatized_tokenized = [extract_bigrams(review) for review in neg_lemmatized_tokenized]
    neg_trigrams_lemmatized_tokenized = [extract_trigrams(review) for review in neg_lemmatized_tokenized]
    neg_bigrams_stemmed_tokenized = [extract_bigrams(review) for review in neg_stemmed_tokenized]
    neg_trigrams_stemmed_tokenized = [extract_trigrams(review) for review in neg_stemmed_tokenized]
    
    # Extract Noun Phrases
    pos_noun_phrases_white_space_tokenized = [extract_noun_phrases(review) for review in pos_white_space_tokenized]
    pos_noun_phrases_lemmatized_tokenized = [extract_noun_phrases(review) for review in pos_lemmatized_tokenized]
    pos_noun_phrases_stemmed_tokenized = [extract_noun_phrases(review) for review in pos_stemmed_tokenized]

    neg_noun_phrases_white_space_tokenized = [extract_noun_phrases(review) for review in neg_white_space_tokenized]
    neg_noun_phrases_lemmatized_tokenized = [extract_noun_phrases(review) for review in neg_lemmatized_tokenized]
    neg_noun_phrases_stemmed_tokenized = [extract_noun_phrases(review) for review in neg_stemmed_tokenized]

    # Save results to JSON files
    FileReader.save_to_json(combine(pos_bigrams_white_space_tokenized, pos_white_space_tokenized), f"{POS_OUTPUT_FOLDER}/bigrams_white_space_tokenized.json")
    FileReader.save_to_json(combine(pos_trigrams_white_space_tokenized, pos_white_space_tokenized), f"{POS_OUTPUT_FOLDER}/trigrams_white_space_tokenized.json")
    FileReader.save_to_json(combine(pos_bigrams_lemmatized_tokenized, pos_lemmatized_tokenized), f"{POS_OUTPUT_FOLDER}/bigrams_lemmatized_tokenized.json")
    FileReader.save_to_json(combine(pos_trigrams_lemmatized_tokenized, pos_lemmatized_tokenized), f"{POS_OUTPUT_FOLDER}/trigrams_lemmatized_tokenized.json")
    FileReader.save_to_json(combine(pos_bigrams_stemmed_tokenized, pos_stemmed_tokenized), f"{POS_OUTPUT_FOLDER}/bigrams_stemmed_tokenized.json")
    FileReader.save_to_json(combine(pos_trigrams_stemmed_tokenized, pos_stemmed_tokenized), f"{POS_OUTPUT_FOLDER}/trigrams_stemmed_tokenized.json")

    FileReader.save_to_json(pos_noun_phrases_white_space_tokenized, f"{POS_OUTPUT_FOLDER}/noun_phrases_white_space_tokenized.json")
    FileReader.save_to_json(pos_noun_phrases_lemmatized_tokenized, f"{POS_OUTPUT_FOLDER}/noun_phrases_lemmatized_tokenized.json")
    FileReader.save_to_json(pos_noun_phrases_stemmed_tokenized, f"{POS_OUTPUT_FOLDER}/noun_phrases_stemmed_tokenized.json")

    FileReader.save_to_json(combine(neg_bigrams_white_space_tokenized, neg_white_space_tokenized), f"{NEG_OUTPUT_FOLDER}/bigrams_white_space_tokenized.json")
    FileReader.save_to_json(combine(neg_trigrams_white_space_tokenized, neg_white_space_tokenized), f"{NEG_OUTPUT_FOLDER}/trigrams_white_space_tokenized.json")
    FileReader.save_to_json(combine(neg_bigrams_lemmatized_tokenized, neg_lemmatized_tokenized), f"{NEG_OUTPUT_FOLDER}/bigrams_lemmatized_tokenized.json")
    FileReader.save_to_json(combine(neg_trigrams_lemmatized_tokenized, neg_lemmatized_tokenized), f"{NEG_OUTPUT_FOLDER}/trigrams_lemmatized_tokenized.json")
    FileReader.save_to_json(combine(neg_bigrams_stemmed_tokenized, neg_stemmed_tokenized), f"{NEG_OUTPUT_FOLDER}/bigrams_stemmed_tokenized.json")
    FileReader.save_to_json(combine(neg_trigrams_stemmed_tokenized, neg_stemmed_tokenized), f"{NEG_OUTPUT_FOLDER}/trigrams_stemmed_tokenized.json")

    FileReader.save_to_json(neg_noun_phrases_white_space_tokenized, f"{NEG_OUTPUT_FOLDER}/noun_phrases_white_space_tokenized.json")
    FileReader.save_to_json(neg_noun_phrases_lemmatized_tokenized, f"{NEG_OUTPUT_FOLDER}/noun_phrases_lemmatized_tokenized.json")
    FileReader.save_to_json(neg_noun_phrases_stemmed_tokenized, f"{NEG_OUTPUT_FOLDER}/noun_phrases_stemmed_tokenized.json")

    print("completed phrase extraction")

# Entry point of the script
if __name__ == "__main__":
    process_phrases()
