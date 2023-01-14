import glob
import os
import pickle
import pprint
from random import random

import numpy as np
import scipy
from scipy.sparse import csc_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score

from RandomClassification import count_files_dir,count_folders_dir, MyTokenizer
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

    global clf, Xtest, Ytest, Ypred, AllGoodSamples, AllMalSamples, FV_original, features_names_arr, features_vocabulary

    count_good_samples = count_folders_dir('.\\AllGoodSamples')
    count_mal_samples = count_folders_dir('.\\AllMalSamples')

    AllGoodSamples = glob.glob(os.path.join('.\\AllGoodSamples\\AllGoodSamples_%s' % count_good_samples, '*txt'))
    AllMalSamples = glob.glob(os.path.join('.\\AllMalSamples\\AllMalSamples_%s' % count_mal_samples, '*txt'))

    count_all = count_files_dir(".\\Clf_storages")

    clf_path = '.\\Clf_storages\\model_%s.pkl' % count_all
    Xtest_path = '.\\Xtests\\Xtest_%s.pkl' % count_all
    Ytest_path = '.\\Ytests\\Ytest_%s.pkl' % count_all
    Ypred_path = '.\\Ypreds\\Ypred_%s.pkl' % count_all
    FV_original_path = '.\\FVs\\FV_%s.pkl' % count_all
    features_names_arr_path = '.\\FEATURES_NAMES\\FEATURE_NAME_%s.pkl' % count_all
    features_vocabulary_path = '.\\FEATURES_VOCABULARYS\\FEATURE_VOCABULARY_%s.pkl' % count_all

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

    f_features_names_arr = open(features_names_arr_path, 'rb')
    features_names_arr = pickle.load(f_features_names_arr)

    f_features_vocabulary = open(features_vocabulary_path, 'rb')
    features_vocabulary = pickle.load(f_features_vocabulary)


# def return_sparse1():
#     FV_good = TF(input='filename', lowercase=False, token_pattern=None,
#                                 tokenizer=MyTokenizer, binary=False, dtype=np.float64)
#
#     FV_mal = TF(input='filename', lowercase=False, token_pattern=None,
#                                tokenizer=MyTokenizer, binary=False, dtype=np.float64)
#
#
#
#     print(AllGoodSamples)
#
#     good_sparse = FV_good.fit_transform(AllGoodSamples)
#     mal_sparse = FV_mal.fit_transform(AllMalSamples)
#
#     dense1 = good_sparse.todense()
#     dense2 = mal_sparse.todense()
#
#     feature_names1 = FV_good.get_feature_names()
#     feature_names2 = FV_mal.get_feature_names()
#
#     value_good = good_sparse.toarray()[0]
#     value_mal = mal_sparse.toarray()[0]
#
#
#     print ("kdfpakjdnmflaksdmfakl;sdf;alsdkf;alkdf;alsdkf;alsdk;alsdkf;alsdkf;alsdkf;alsdkfa;sdlkfa;sdlkf")
#
#     #pprint.pprint(tfidf_dict_good)
#     #pprint.pprint(feature_names1)
#
#     tfidf_dict_good = dict(zip(feature_names1, value_good))
#     tfidf_dict_mal = dict(zip(feature_names2, value_mal))
#
#
#     return tfidf_dict_good, tfidf_dict_mal


def return_sparse():
    global FV_good, FV_mal

    FV_good = TF(input='filename', lowercase=False, token_pattern=None,
                                tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    FV_mal = TF(input='filename', lowercase=False, token_pattern=None,
                               tokenizer=MyTokenizer, binary=False, dtype=np.float64)


    good_sparse = FV_good.fit_transform(AllGoodSamples)
    mal_sparse = FV_mal.fit_transform(AllMalSamples)
    #pprint.pprint(FV_mal.vocabulary_)
    return good_sparse, mal_sparse


# def set_Xtest_sparse(feature, FV):
#     return Xtest.toarray()[FV.vocabulary_[feature]]
#
#
# def reformat_Xtest(tfidf_dict_good, tfidf_dict_mal):
#     for f_name in tfidf_dict_mal:
#         num = set_Xtest_sparse(f_name, FV_mal)
#         if num<= 0.1 and num is not None:
#             Xtest[num] = random.uniform(0.5, 1)


def belong_feature(feature_name):
    flag_good=False
    flag_mal=False
    if FV_good.vocabulary_.get(feature_name) is not None:
        flag_good=True

    if FV_mal.vocabulary_.get(feature_name) is not None:
        flag_mal=True

    return flag_good, flag_mal


def set_sparse_column_to_zero(matrix, col_index):
    # Make sure the matrix is in the CSC format
    matrix = csc_matrix(matrix)
    # Set the values of non-zero elements in the column to zero
    matrix.data[matrix.indptr[col_index]:matrix.indptr[col_index + 1]] = 0
    return matrix


def sum_sparse_column(matrix, col_index):
    # Make sure the matrix is in the CSC format
    matrix = csc_matrix(matrix)
    # Get the indices of non-zero elements in the column
    col_indices = matrix.indices[matrix.indptr[col_index]:matrix.indptr[col_index + 1]]
    # Get the values of non-zero elements in the column
    col_values = matrix.data[matrix.indptr[col_index]:matrix.indptr[col_index + 1]]
    # Sum the values
    return sum(col_values)


def get_feature_avg(feature_name, sparse_matrix, FV):
    sum = 0.0
    feature_idx = FV.vocabulary_[feature_name]

    sparse_matrix_dense = sparse_matrix.todense()

    sum_sparse_column(sparse_matrix_dense, feature_idx)
    return sum/sparse_matrix.shape[0]


def set_Xtest(feature_name):
    sparse_good, sparse_mal = return_sparse()

    flag_good, flag_mal = belong_feature(feature_name)

    global Xtest
    if flag_mal:
        FV_original_voc = FV_original.vocabulary_
        feature_idx = FV_original_voc[feature_name]
        num = get_feature_avg(feature_name, sparse_mal, FV_mal)
        if num > 0.4:

            Xtest = set_sparse_column_to_zero(Xtest, feature_idx)
            pprint.pprint(Xtest)

    # if flag_good and flag_mal:
    #     return
    #
    # elif flag_good:
    #     return
    #
    # elif flag_mal:
    #     FV_original_voc = FV_original.vocabulary_
    #     feature_idx = FV_original_voc[feature_name]
    #     num = get_feature_avg(feature_name, sparse_mal, FV_mal)
    #     if num > 0.4:
    #
    #         Xtest = set_sparse_column_to_zero(Xtest, feature_idx)
    #         pprint.pprint(Xtest)


def set_all_feature():
    print ("adlkjfhadlfhasdkfjhadlkfjhasdlkfjahsdlkjashld!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    i=0
    for feature_name in FV_original.get_feature_names():
        if i == 200:
            return Xtest
        set_Xtest(feature_name)
        i+=1
    return Xtest



def wrrap():
    get_last_try()

    #reformat_Xtest(dict_good, dict_mal)
    #new_Xtest = Xtest_attack()

    new_Ypred = clf.predict(set_all_feature())

    before = accuracy_score(Ytest, Ypred)
    after = accuracy_score(Ytest, new_Ypred)

    print "Accuracy before attack: ", before
    print(metrics.classification_report(Ytest, Ypred, labels=[1, -1], target_names=['Malware', 'Goodware']))

    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! After Attack !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    print "Accuracy after attack: ", after
    print(metrics.classification_report(Ytest, new_Ypred, labels=[1, -1],
                                        target_names=['Malware', 'Goodware']))






if __name__ == '__main__':
    wrrap()
