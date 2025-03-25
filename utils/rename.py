import os
import re


folder_path = "/gdata1/gaocs/Data_FCM_NQ/dinov2/seg/hyperprior/training_log/trunl-530.9767_trunh103.2168_kmeans10_bitdepth8"


pattern = re.compile(r"(.*)_epoch(\d+)_lr0\.0001(.*)")


for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        new_filename = f"{match.group(1)}_epochs{match.group(2)}_lr0.0001{match.group(3)}"
        
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("Batch rename completed!")


