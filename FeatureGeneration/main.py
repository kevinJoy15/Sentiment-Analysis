# used to run everything in one go
import os
import shutil
import Tokenisation
import PhraseExtraction
import Boosting
import Normalisation
import FeatureSelection
import Splitfiles
import NaiveBayesScratch
import FileReader

input_folder = "coursework/dataset/trainDataset"
eval_folder = "coursework/dataset/evaluationDataset"
test_folder = "coursework/dataset/testDataset"

new_path = shutil.copytree('coursework/dataset/data', 'coursework/dataset/trainDataset', dirs_exist_ok=True)

FileReader.clear_folder("coursework/dataset/evaluationDataset")
FileReader.clear_folder("coursework/dataset/testDataset")
FileReader.clear_folder("coursework/dataset/trainDataset")

Splitfiles.split_reviews(input_folder, eval_folder, test_folder, 300)

print("number of files in pos train folder", len(os.listdir(f"{input_folder}/pos")))
print("number of files in neg train folder", len(os.listdir(f"{input_folder}/neg")))
print("number of files in eval folder", len(os.listdir(eval_folder))) 
print("number of files in test folder", len(os.listdir(test_folder))) 

Tokenisation.process_dataset(frequency_threshold=5)
PhraseExtraction.process_phrases()
Boosting.process_with_boosting(1.5)
Normalisation.process_normalization()
FeatureSelection.main()
NaiveBayesScratch.main()
