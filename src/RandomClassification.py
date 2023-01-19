import pickle
import pprint
import shutil

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
import os, sys, glob
from random import randint
import logging
from time import time
import warnings


# filter out warnings about deprecated modules
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)
# logging level
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def MyTokenizer(Str):
    return Str.split()


def count_files_dir(path_dir):
    """
    input:
        path_dir: (string) path to directory

    output:
        count: (int) how many files are in this directory
    """
    count = 0
    # Iterate directory
    for path in os.listdir(path_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_dir, path)):
            count += 1

    return count


def count_folders_dir(directory_path):
    return len([f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))])


def dump_argument(path, name, obj):
    if not os.path.exists(path):
        os.mkdir(path)
    count_TF = count_files_dir(path) + 1
    pickle.dump(obj, open('%s\\%s_%s.pkl' % (path, name, count_TF), 'wb'))

def RandomClassification(MalwareCorpus, GoodwareCorpus, TestSize, NumFeaturesToBeSelected, FeatureOption):
    '''
    Train a classifier for classifying malwares and goodwares using Random Forest technique
    Compute the prediction accuracy and f1 score of the classifier

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param Float TestSize: test set split (default is 0.2 for testing and 0.8 for training) - won't be set
    :param integer NumFeaturesToBeSelected: number of top features to select
    :param Boolean FeatureOption: False
    '''

    # Step 1: Getting the malware and goodware txt files
    Logger.debug("Loading positive and negative samples")
    AllMalSamples = glob.glob(os.path.join(MalwareCorpus, '*txt'))
    AllGoodSamples = glob.glob(os.path.join(GoodwareCorpus, '*txt'))
    print(AllGoodSamples)
    Logger.info("All Samples loaded")

    # Step 2: Creating feature vector
    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None,
                           tokenizer=MyTokenizer, binary=FeatureOption, dtype=np.float64)

    Logger.info("#################################################################################")
    Logger.info(FeatureVectorizer)

    Logger.info("#################################################################################")
    dump_argument(".\\FVs_change", "FV_change", FeatureVectorizer)

    X = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)

    feature_names_array = FeatureVectorizer.get_feature_names()
    feature_vocabulary_array = FeatureVectorizer.vocabulary_
    dump_argument(".\\FEATURES_NAMES", "FEATURE_NAME", feature_names_array)
    dump_argument(".\\FEATURES_VOCABULARYS", "FEATURE_VOCABULARY", feature_vocabulary_array)

    #pprint.pprint(FeatureVectorizer.get_feature_names())
    FeatureVectorizer_change = TF(input='filename', lowercase=False, token_pattern=None,
                           tokenizer=MyTokenizer, binary=FeatureOption, dtype=np.float64)
    dump_argument(".\\FVs", "FV", FeatureVectorizer)

    Logger.info("#################################################################################")
    Logger.info(X)
    Logger.info('1231232132132132132132132132132132132132132132132132132')
    Logger.info(len(FeatureVectorizer.get_feature_names()))
    Logger.info(len(AllMalSamples))
    Logger.info(len(AllGoodSamples))
    Logger.info("#################################################################################")

    # Label malware as 1 and goodware as -1
    MalLabels = np.ones(len(AllMalSamples))
    GoodLabels = np.empty(len(AllGoodSamples))

    if not os.path.exists(".\\AllMalSamples"):
        os.mkdir(".\\AllMalSamples")
    count_AllMalSamples = count_folders_dir(".\\AllMalSamples") + 1
    os.mkdir(".\\AllMalSamples\\AllMalSamples_%s" % count_AllMalSamples)
    count_files = 1
    for txt_file in AllMalSamples:
        shutil.copy(txt_file, '.\\AllMalSamples\\AllMalSamples_%s\\good_txt_file_%s.txt' % (count_AllMalSamples, count_files))
        count_files += 1

    if not os.path.exists(".\\AllGoodSamples"):
        os.mkdir(".\\AllGoodSamples")
    count_AllGoodSamples = count_folders_dir(".\\AllGoodSamples") + 1
    os.mkdir(".\\AllGoodSamples\\AllGoodSamples_%s" % count_AllGoodSamples)
    count_files = 1
    for txt_file in AllGoodSamples:
        shutil.copy(txt_file,'.\\AllGoodSamples\\AllGoodSamples_%s\\mal_txt_file_%s.txt' % (count_AllGoodSamples, count_files))
        count_files += 1

    if not os.path.exists(".\\AllMalSamples_change"):
        os.mkdir(".\\AllMalSamples_change")
    count_AllMalSamples = count_folders_dir(".\\AllMalSamples_change") + 1
    os.mkdir(".\\AllMalSamples_change\\AllMalSamples_change_%s" % count_AllMalSamples)
    count_files = 1
    for txt_file in AllMalSamples:
        shutil.copy(txt_file, '.\\AllMalSamples_change\\AllMalSamples_change_%s\\good_txt_file_%s.txt' % (count_AllMalSamples, count_files))
        count_files += 1

    if not os.path.exists(".\\AllGoodSamples_change"):
        os.mkdir(".\\AllGoodSamples_change")
    count_AllGoodSamples = count_folders_dir(".\\AllGoodSamples_change") + 1
    os.mkdir(".\\AllGoodSamples_change\\AllGoodSamples_change_%s" % count_AllGoodSamples)
    count_files = 1
    for txt_file in AllGoodSamples:
        shutil.copy(txt_file,'.\\AllGoodSamples_change\\AllGoodSamples_change_%s\\mal_txt_file_%s.txt' % (count_AllGoodSamples, count_files))
        count_files += 1

    GoodLabels.fill(-1)
    Y = np.concatenate((MalLabels, GoodLabels), axis=0)
    Logger.info("Label array - generated")

    # Step 3: Split all samples into training and test set
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y,
                                                    test_size=0.2, random_state=randint(0, 100))
    Logger.debug("Test set split = %s", TestSize)

    Features = FeatureVectorizer.get_feature_names()
    # her i started to get the features:
    Logger.info("################################################################")
    bound = 0
    for i in Features:
        if bound > 20:
            break
        Logger.info(i)
        bound += 1

    Logger.info("################################################################")

    Logger.info("Total number of features: {} ".format(len(Features)))

    if len(Features) > NumFeaturesToBeSelected:
        # with feature selection
        Logger.info("Gonna select %s features", NumFeaturesToBeSelected)
        FSAlgo = SelectKBest(chi2, k=NumFeaturesToBeSelected)

        XTrain = FSAlgo.fit_transform(XTrain, YTrain)
        dump_argument(".\\FSAlgos", "FSAlgo", FSAlgo)
        print XTest.shape
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        XTest = FSAlgo.transform(XTest)

    Logger.info("Gonna perform classification with C-RandomForest")

    # Step 4: model selection through cross validation
    # Assuming RandomForest is the only classifier we are gonna try, we will set the n_estimators parameter as follows.
    Parameters = {'n_estimators': [10, 50, 100, 200, 500, 1000],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    Clf = GridSearchCV(RandomForestClassifier(), Parameters, cv=5, scoring='f1', n_jobs=-1)
    RFmodels = Clf.fit(XTrain, YTrain)
    BestModel = RFmodels.best_estimator_
    Logger.info('CV done - Best model selected: {}'.format(BestModel))
    # Best model is chosen through 5-fold cross validation and stored in the variable: RFmodels

    # save the classifier and the Xtest, Ytest and the clf

    dump_argument(".\\Clf_storages", "model", RFmodels)
    dump_argument(".\\Xtests", "Xtest", XTest)
    dump_argument(".\\Ytests", "Ytest", YTest)


    # Step 5: Evaluate the best model on test set
    Ypred = RFmodels.predict(XTest)
    dump_argument(".\\Ypreds", "Ypred", Ypred)

    Accuracy = accuracy_score(YTest, Ypred)

    Logger.info("#################################################################################")
    Logger.info("and they are the predictions answer")
    for i in Ypred:
        Logger.info(i)
    Logger.info("and they are the true answer")
    for i in YTest:
        Logger.info(i)

    Logger.info("#################################################################################")

    print "Test Set Accuracy = ", Accuracy
    print(metrics.classification_report(YTest, Ypred, labels=[1, -1], target_names=['Malware', 'Goodware']))
