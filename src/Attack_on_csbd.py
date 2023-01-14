import glob
import os
import pickle
from random import random

import numpy as np
import scipy
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

    global clf, Xtest, Ytest, Ypred, AllGoodSamples, AllMalSamples

    count_good_samples = count_folders_dir('.\\AllGoodSamples')
    count_mal_samples = count_folders_dir('.\\AllMalSamples')

    AllGoodSamples = glob.glob(os.path.join('.\\AllGoodSamples\\AllGoodSamples_%s' % count_good_samples, '*txt'))
    AllMalSamples = glob.glob(os.path.join('.\\AllMalSamples\\AllMalSamples_%s' % count_mal_samples, '*txt'))

    count_clf = count_files_dir(".\\Clf_storage")
    count_Xtests = count_files_dir(".\\Xtests")
    count_Ytests = count_files_dir(".\\Ytests")
    count_Ypreds = count_files_dir(".\\Ypreds")

    clf_path = '.\\Clf_storage\\model_%s.pkl' % count_clf
    Xtest_path = '.\\Xtests\\Xtest_%s.pkl' % count_Xtests
    Ytest_path = '.\\Ytests\\Ytest_%s.pkl' % count_Ytests
    Ypred_path = '.\\Ypreds\\Ypred_%s.pkl' % count_Ypreds

    f_clf = open(clf_path, 'rb')
    clf = pickle.load(f_clf)

    f_Xtest = open(Xtest_path, 'rb')
    Xtest = pickle.load(f_Xtest)

    f_Ytest = open(Ytest_path, 'rb')
    Ytest = pickle.load(f_Ytest)

    f_Ypred = open(Ypred_path, 'rb')
    Ypred = pickle.load(f_Ypred)


def return_sparse1():
    FV_good = TF(input='filename', lowercase=False, token_pattern=None,
                                tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    FV_mal = TF(input='filename', lowercase=False, token_pattern=None,
                               tokenizer=MyTokenizer, binary=False, dtype=np.float64)



    print(AllGoodSamples)

    good_sparse = FV_good.fit_transform(AllGoodSamples)
    mal_sparse = FV_mal.fit_transform(AllMalSamples)

    dense1 = good_sparse.todense()
    dense2 = mal_sparse.todense()

    feature_names1 = FV_good.get_feature_names()
    feature_names2 = FV_mal.get_feature_names()

    value_good = good_sparse.toarray()[0]
    value_mal = mal_sparse.toarray()[0]


    print ("kdfpakjdnmflaksdmfakl;sdf;alsdkf;alkdf;alsdkf;alsdk;alsdkf;alsdkf;alsdkf;alsdkfa;sdlkfa;sdlkf")

    #pprint.pprint(tfidf_dict_good)
    #pprint.pprint(feature_names1)

    tfidf_dict_good = dict(zip(feature_names1, value_good))
    tfidf_dict_mal = dict(zip(feature_names2, value_mal))


    return tfidf_dict_good, tfidf_dict_mal


def return_sparse():
    global FV_good, FV_mal

    FV_good = TF(input='filename', lowercase=False, token_pattern=None,
                                tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    FV_mal = TF(input='filename', lowercase=False, token_pattern=None,
                               tokenizer=MyTokenizer, binary=False, dtype=np.float64)

    good_sparse = FV_good.fit_transform(AllGoodSamples)
    mal_sparse = FV_mal.fit_transform(AllMalSamples)

    arr_good = FV_good.get_feature_names()
    arr_mal = FV_mal.get_feature_names()

    return arr_good, arr_mal


def set_Xtest_sparse(feature, FV):
    return Xtest.toarray()[FV.vocabulary_[feature]]


def reformat_Xtest(tfidf_dict_good, tfidf_dict_mal):
    for f_name in tfidf_dict_mal:
        num = set_Xtest_sparse(f_name, FV_mal)
        if num<= 0.1 and num is not None:
            Xtest[num] = random.uniform(0.5, 1)


def get_index(feature_name, sparse_matrix):
    sparse_matrix.toarray()
    #for i in spar

def fun(feature_name, feature_value):

    for doc in Xtest.len():
        Xtest[doc][0].set




# def Xtest_attack():
#     # for i, j in zip(*Xtest.nonzero()):
#     #     Xtest[i, j] = Xtest[i, j] * random()
#     # return Xtest



def wrrap():
    get_last_try()
    dict_good, dict_mal = return_sparse()
    reformat_Xtest(dict_good, dict_mal)
    #new_Xtest = Xtest_attack()
    new_Ypred = clf.predict(Xtest)

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
