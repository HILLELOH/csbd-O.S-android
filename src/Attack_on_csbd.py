import pickle

from sklearn import metrics
from sklearn.metrics import accuracy_score

from RandomClassification import count_dir
import logging

# logging level
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def get_last_try():
    """
    return the loaded clf, Xtest, Ytest, Ypred
    from the pickle files that "Random Classification" write
    """

    global clf, Xtest, Ytest, Ypred
    Logger.info("isdjhfopisjfoiajsd")

    count_clf = count_dir(".\\Clf_storage")
    count_Xtests = count_dir(".\\Xtests")
    count_Ytests = count_dir(".\\Ytests")
    count_Ypreds = count_dir(".\\Ypreds")

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


def Xtest_attack():
    return Xtest


def wrrap():
    get_last_try()
    new_Xtest = Xtest_attack()
    after_attack_prediction = clf.predict(new_Xtest)
    before = accuracy_score(Ytest, Ypred)
    after = accuracy_score(Ytest, after_attack_prediction)

    print "Accuracy before attack: ", before
    print(metrics.classification_report(Ytest, Ypred, labels=[1, -1], target_names=['Malware', 'Goodware']))

    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! After Attack !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    print "Accuracy after attack: ", after
    print(metrics.classification_report(Ytest, after_attack_prediction, labels=[1, -1],
                                        target_names=['Malware', 'Goodware']))


if __name__ == '__main__':
    wrrap()
