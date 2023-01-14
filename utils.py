import os, json

def write_file(data, filepath, filename):
    if os.path.isdir(filepath) == False:
        try:
            os.makedirs(filepath)
        except:
            print('Error making directory')

    if '.csv' in filename:
        data.to_csv(os.path.join(filepath, filename))
    else:
        with open(os.path.join(filepath, filename), "w") as outfile:
            outfile.write(data)
