import os
import random

def delete_half_of_files(folder_path):
    # Get list of all files in the directory
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Calculate how many files to delete (half of the total)
    num_files_to_delete = len(all_files) // 2
    
    # Randomly select files to delete
    files_to_delete = random.sample(all_files, num_files_to_delete)
    
    # Delete selected files
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# Usage example
# folder_path = '/dddsdsds'
# delete_half_of_files(folder_path)