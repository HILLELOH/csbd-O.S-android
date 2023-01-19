import os
import shutil


def clean_dir(directory):
    # remove all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clean_all(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clean_dir(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # remove the directory
    #os.rmdir(directory)

if __name__ == '__main__':
    clean_all(".\\Clf_storages")
    clean_all(".\\Xtests")
    clean_all(".\\Ytests")
    clean_all(".\\Ypreds")
    clean_all(".\\FVs")
    clean_all(".\\FVs_change")
    clean_all(".\\AllGoodSamples")
    clean_all(".\\AllMalSamples")
    clean_all(".\\AllGoodSamples_change")
    clean_all(".\\AllMalSamples_change")
    clean_all(".\\FEATURES_NAMES")
    clean_all(".\\FEATURES_VOCABULARYS")
    clean_all(".\\FSAlgos")