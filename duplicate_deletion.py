## Delete duplicated images. If the refresh interval in "download_TpDne.py" is set too low, duplicate might be
## dowloaded.

import hashlib
import os

# Hash all non-replicate files
def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()

# Get Folder Path
os.getcwd()
os.chdir(r'./Test_Images')
os.getcwd()

# Create the list of all files
file_list = os.listdir()
print(len(file_list))

# Find Duplicates
duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))

print(duplicates)
print(len(duplicates))

# Delete Duplicates
for index in duplicates:
    os.remove(file_list[index[0]])