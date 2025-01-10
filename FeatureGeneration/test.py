from nltk.corpus import wordnet as wn

token = "happy"
synsets = wn.synsets(token)
print(synsets)
