import glob
import os
import pickle
import pprint
from random import random, randint

import numpy as np
import scipy
from scipy.sparse import csc_matrix
from scipy.stats import chi2
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score

from RandomClassification import *
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import logging
import warnings

# filter out warnings about deprecated modules
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# logging level
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def get_last_try():
    """
    return the loaded clf, Xtest, Ytest, Ypred
    from the pickle files that "Random Classification" write
    """

    global clf, Xtest, Ytest, Ypred, AllGoodSamples, AllMalSamples, AllGoodSamples_change, AllMalSamples_change, FV_original, \
    FV_change, Xtest_features_names, Xtest_features_vocabulary, FSAlgo

    count_good_samples = count_folders_dir('.\\AllGoodSamples')
    count_mal_samples = count_folders_dir('.\\AllMalSamples')

    AllGoodSamples = glob.glob(os.path.join('.\\AllGoodSamples\\AllGoodSamples_%s' % count_good_samples, '*txt'))
    AllMalSamples = glob.glob(os.path.join('.\\AllMalSamples\\AllMalSamples_%s' % count_mal_samples, '*txt'))

    AllGoodSamples_change = glob.glob(os.path.join('.\\AllGoodSamples_change\\AllGoodSamples_change_%s' % count_good_samples, '*txt'))
    AllMalSamples_change = glob.glob(os.path.join('.\\AllMalSamples_change\\AllMalSamples_change_%s' % count_mal_samples, '*txt'))

    count_all = count_files_dir(".\\Clf_storages")

    clf_path = '.\\Clf_storages\\model_%s.pkl' % count_all
    Xtest_path = '.\\Xtests\\Xtest_%s.pkl' % count_all
    Ytest_path = '.\\Ytests\\Ytest_%s.pkl' % count_all
    Ypred_path = '.\\Ypreds\\Ypred_%s.pkl' % count_all
    FV_original_path = '.\\FVs\\FV_%s.pkl' % count_all
    FV_change_path = '.\\FVs_change\\FV_change_%s.pkl' % count_all
    features_names_arr_path = '.\\FEATURES_NAMES\\FEATURE_NAME_%s.pkl' % count_all
    features_vocabulary_path = '.\\FEATURES_VOCABULARYS\\FEATURE_VOCABULARY_%s.pkl' % count_all
    FSAlgo_path = '.\\FSAlgos\\FSAlgo_%s.pkl' % count_all

    f_clf = open(clf_path, 'rb')
    clf = pickle.load(f_clf)

    f_Xtest = open(Xtest_path, 'rb')
    Xtest = pickle.load(f_Xtest)

    f_Ytest = open(Ytest_path, 'rb')
    Ytest = pickle.load(f_Ytest)

    f_Ypred = open(Ypred_path, 'rb')
    Ypred = pickle.load(f_Ypred)

    f_FV_original = open(FV_original_path, 'rb')
    FV_original = pickle.load(f_FV_original)

    f_FV_change = open(FV_change_path, 'rb')
    FV_change = pickle.load(f_FV_change)

    f_features_names_arr = open(features_names_arr_path, 'rb')
    Xtest_features_names = pickle.load(f_features_names_arr)

    f_features_vocabulary = open(features_vocabulary_path, 'rb')
    Xtest_features_vocabulary = pickle.load(f_features_vocabulary)

    f_FSAlgo_path = open(FSAlgo_path, 'rb')
    FSAlgo = pickle.load(f_FSAlgo_path)

def return_sparse():
    global FV_good, FV_mal

    FV_good = TF(input='filename', lowercase=False, token_pattern=None,
                 tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    FV_mal = TF(input='filename', lowercase=False, token_pattern=None,
                tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    good_sparse = FV_good.fit_transform(AllGoodSamples)
    mal_sparse = FV_mal.fit_transform(AllMalSamples)
    # pprint.pprint(FV_mal.vocabulary_)
    # return good_sparse, mal_sparse
    return FV_mal, FV_good

def set_sparse_column_to_zero(matrix, col_index):
    # Make sure the matrix is in the CSC format
    matrix = csc_matrix(matrix)
    # Set the values of non-zero elements in the column to zero
    matrix.data[matrix.indptr[col_index]:matrix.indptr[col_index + 1]] = 0
    return matrix


def get_features_avg(sparse_matrix):
    sum_array = sparse_matrix.sum(axis=0)
    length = sparse_matrix.shape[0]
    print length
    for sum in sum_array:
        sum /= length

    return sum_array


def get_feature_sets():
    return_sparse()
    good_dict_feature = FV_good.vocabulary_
    mal_dict_feature = FV_mal.vocabulary_

    avg_arr = np.ravel(get_features_avg(Xtest))
    global good_feature_set, mal_feature_set
    good_feature_set = {}
    mal_feature_set = {}

    for feature in Xtest_features_names:

        if Xtest_features_vocabulary[feature] >= 5000:
            break

        flag_in_good, flag_in_mal = feature in FV_good.get_feature_names(), feature in FV_mal.get_feature_names()

        if flag_in_good:
            average_feature = avg_arr[good_dict_feature[feature]]

        if flag_in_mal:

            average_feature = avg_arr[Xtest_features_vocabulary[feature]]
            if average_feature > 0.2:
                mal_feature_set[feature] = Xtest_features_vocabulary[feature], average_feature
                mal_feature_set[Xtest_features_vocabulary[feature]] = feature, average_feature
                print "ok"

        # else:
        #     Logger.info("The feature isn't in good, mal!")

    return good_feature_set, mal_feature_set


def get_feature_name(idx):
    for key in Xtest_features_vocabulary.keys():
        if key == idx:
            return Xtest_features_names[idx]

    return None


def change_random_value():
    #good_feature_set, mal_feature_set = get_feature_sets()
    #FV_mal, FV_good = return_sparse()

    for i, j in zip(*Xtest.nonzero()):
        Xtest[i, j] = Xtest[i, j] * (0.01 * random())
        # if get_feature_name(j) in FV_mal.get_feature_names():
        #
        #
        # if get_feature_name(j) in FV_good.get_feature_names():
        #     Xtest[i, j] = Xtest[i, j] * 0.01

    return Xtest




def wrrap():
    get_last_try()
    print Xtest.shape
    # reformat_Xtest(dict_good, dict_mal)
    new_Xtest = change_random_value()


    # new_Xtest = change_txt()
    # print new_Xtest.shape
    #
    # new_Xtest = FSAlgo.transform(new_Xtest)

    p = clf.predict(new_Xtest)

    #before = accuracy_score(Ytest, Ypred)
    before = accuracy_score(Ytest, Ypred)
    after = accuracy_score(Ytest, p)

    if before - after < 0.045:
        print "try againnnnnnnnnnnnnnnnnnnn"
        wrrap()



    else:
        #count_old_Xtest = count_files_dir(".\\Xtests_attacked")
        dump_argument(".\\Xtests_attacked", "Xtest_attacked", new_Xtest)
        print "............................................................................................"
        print "Accuracy before attack: ", before
        print(metrics.classification_report(Ytest, Ypred, labels=[1, -1], target_names=['Malware', 'Goodware']))

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! After Attack !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        print "Accuracy after attack: ", after
        print(metrics.classification_report(Ytest, p, labels=[1, -1],
                                            target_names=['Malware', 'Goodware']))
        print "............................................................................................"


if __name__ == '__main__':
    wrrap()

    #replace_strings_in_files('C:\\Users\\hille\\PycharmProjects\\csbd-O.S-android\\src\\AllGoodSamples_change\\AllGoodSamples_change_1\\try', "is", "okdfnasldkfn")