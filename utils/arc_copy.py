import os
import shutil

def copy_folders_based_on_file(file_path, source_dir, destination_dir):
    """
    Copy specific subfolders and their files from a source directory to a destination directory 
    based on the content of a text file.

    :param file_path: Path to the text file containing folder and file names.
    :param source_dir: Path to the source directory containing all subfolders.
    :param destination_dir: Path to the destination directory to store the copied subfolders.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Read the file and process each line
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_name = line.split()[0]  # Split folder and file name
            # folder_name = folder_name[2:-1]; file_name = file_name[2:-2]
            print(file_name)
            
            source_folder_path = os.path.join(source_dir, file_name[:-4]+'.npy')
            destination_folder_path = os.path.join(destination_dir, file_name[:-4]+'.npy')
                
            # Copy the specific file if it exists
            if os.path.exists(source_folder_path):
                shutil.copy2(source_folder_path, destination_folder_path)

# Main function to run the script
if __name__ == "__main__":
    # text_file_path = "/gdata1/gaocs/FCM_LM_Test_Dataset/llama3/csr/source/arc_challenge_test_longest100_shape.txt"  # Replace with the actual path to the text file
    # source_directory = "/gdata1/gaocs/FCM_LM_Test_Dataset/llama3/csr/feature"  # Replace with the actual path to the source directory
    # destination_directory = "/gdata1/gaocs/FCM_LM_Train_Data/llama3/csr/org_feat/test"  # Replace with the desired destination directory
    
    # copy_folders_based_on_file(text_file_path, source_directory, destination_directory)


    text_file_path = "/gdata1/gaocs/FCM_LM_Test_Dataset/sd3/tti/source/captions_val2017_select100.txt"  # Replace with the actual path to the text file
    source_directory = "/gdata1/gaocs/FCM_LM_Test_Dataset/sd3/tti/feature"  # Replace with the actual path to the source directory
    destination_directory = "/gdata1/gaocs/FCM_LM_Train_Data/sd3/tti/org_feat/test"  # Replace with the desired destination directory
    
    copy_folders_based_on_file(text_file_path, source_directory, destination_directory)
