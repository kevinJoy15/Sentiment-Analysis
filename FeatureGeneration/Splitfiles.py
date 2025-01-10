import os
import shutil
import random

def split_reviews(input_folder, eval_folder, test_folder, fixed_number):
    
    # Define paths for `pos` and `neg` subfolders
    pos_folder = os.path.join(input_folder, "pos")
    neg_folder = os.path.join(input_folder, "neg")
        
    # Function to process files
    def process_files(source_folder, target_folder, prefix, fixed_number):
        files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

        random.shuffle(files)
        
        # If the number of files is less than the fixed number, take all the files available
        fixed_number = min(fixed_number, len(files))
        
        selected_files = random.sample(files, fixed_number)

        for f in selected_files:
            new_filename = f"{prefix}_{f}"
            shutil.move(os.path.join(source_folder, f), os.path.join(target_folder, new_filename))
    
    # Process `pos` files for eval and test
    process_files(pos_folder, eval_folder, "pos", fixed_number)
    process_files(pos_folder, test_folder, "pos", fixed_number)
    
    # Process `neg` files for eval and test
    process_files(neg_folder, eval_folder, "neg", fixed_number)
    process_files(neg_folder, test_folder, "neg", fixed_number)
