import os, json
from google.colab import files

def write_file(data, filepath, filename, colab_checkpoint=False):
    if os.path.isdir(filepath) == False:
        try:
            os.makedirs(filepath)
        except:
            print('Error making directory')

    path = os.path.join(filepath, filename)

    if '.csv' in filename:
        data.to_csv(path)
    else:
        with open(path, "w") as outfile:
            outfile.write(data)
    
    if colab_checkpoint==True:
        files.download(path)
