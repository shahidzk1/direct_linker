from __future__ import print_function

import sys
import numpy as np
from functools import cmp_to_key, partial
from sklearn.preprocessing import QuantileTransformer
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
import tensorflow as tf

class RankingCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 min_delta=0,
                 patience=0,
                 cur_epoch=0,
                 best_score=0,
                 best_epoch=0,
                 verbose=0,
                 validation_data=None
                 ):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.verbose = verbose
        self.cur_epoch = cur_epoch
        self.validation_data = validation_data
        self.stop = False

    def on_epoch_end(self, epoch, logs=None):
        x_0, x_1, y_0, y_1, = self.validation_data

        x_test = np.concatenate([x_0, x_1], axis=0)
        x_test_conv = np.expand_dims(x_test, axis=2)
        y_test = np.concatenate([y_0, y_1], axis=0)

        res_test = self.model.predict(
            [x_test, np.zeros(np.shape(x_test)), x_test_conv, np.zeros(np.shape(x_test_conv))],
        )

        fpr, tpr, thresholds = metrics.roc_curve(y_test, res_test, pos_label=1)
        order = np.lexsort((tpr, fpr))
        fpr, tpr = fpr[order], tpr[order]
        auc_test = metrics.auc(fpr, tpr)
        tpr = tpr[fpr > 0]
        fpr = fpr[fpr > 0]
        s0 = np.sqrt(2 * ((tpr + fpr) * np.log((1 + tpr / fpr)) - tpr))
        if self.verbose > 0:
            print('Epoch {} Validation AUC {} S0 {}'.format(self.cur_epoch, auc_test, s0.max()))
        cur_score = auc_test * s0.max() + self.min_delta
        if cur_score > self.best_score:
            self.best_score = auc_test * s0.max()
            self.best_epoch = self.cur_epoch

        if (self.best_epoch + self.patience) <= self.cur_epoch:
            if self.verbose > 0:
                print("Epoch %05d: early stopping " % self.best_epoch)
            self.stop = True

def AMS(estimator, x, y_true, bdt=False):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    y_pred = estimator.predict_proba(x)
    if bdt:
        y_pred = y_pred[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    radicand = 2 *( (tpr+fpr+br) * math.log (1.0 + tpr/(fpr+br)) - tpr)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def read_higgs_data(path_higgs, path_ttbar, test_data=False, luminosity=156000, preprocess_fn=QuantileTransformer(), get_names=False):
    higgs = pd.read_csv(path_higgs)
    ttbar = pd.read_csv(path_ttbar)
    df = pd.concat([higgs, ttbar])
    del df["Unnamed: 0"]
    x_names = set(df.columns) - {'label', 'weight'}

    x = df.loc[:, x_names].values
    y = df.loc[:, 'label'].values
    w = df.loc[:, 'weight'].values * luminosity

    if preprocess_fn != "":
        x = preprocess_fn.fit_transform(x)
    if test_data:
        x = np.append(x[y == 1][:500], x[y == 0][:500], axis=0)
        w = np.append(w[y == 1][:500], w[y == 0][:500], axis=0)
        y = np.append(y[y == 1][:500], y[y == 0][:500], axis=0)
    if get_names:
        return x, y, w, list(df.columns)
    else:
        return x, y, w


def higgs_ranking_metric(estimator, x, y_true, bdt=False, w_0=1, w_1=1, luminosity=156000, b_r=0):
    y_pred = estimator.predict_proba(x)
    if bdt:
        y_pred = y_pred[:, 1]
    w = [w_0 * luminosity if yi == 1 else w_1 * luminosity for yi in y_true]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, sample_weight=w, pos_label=1)
    tpr = tpr[fpr > 0]
    fpr = fpr[fpr > 0]
    order = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[order], tpr[order]
    roc_auc = metrics.auc(fpr, tpr)
    s0 = np.sqrt(2 * ((tpr + fpr + b_r) * np.log((1 + tpr / (fpr + b_r))) - tpr))
    return roc_auc * s0.max()


def plotPRC(estimator, x, y, name, cnn_bdt=False):
    if cnn_bdt:
        dr_predict = estimator.predict_proba(x)[:, 1]
    else:
        dr_predict = estimator.predict_proba(x)

    precision, recall, _ = precision_recall_curve(y, dr_predict, pos_label=1)

    np.save(str(name) + "_precision", precision)
    np.save(str(name) + "_recall", recall)

    prc_auc = metrics.auc(recall, precision)

    plt.plot(recall, precision, lw=1, label='PRC (area = %0.4f)' % (prc_auc))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig("prc_auc_ranker_cls.pdf")
    plt.close()


def plotAUC(estimator, x, y, name, cnn_bdt=False):
    if cnn_bdt:
        dr_predict = estimator.predict_proba(x)[:, 1]
    else:
        dr_predict = estimator.predict_proba(x)

    fpr, tpr, thresholds = metrics.roc_curve(y, dr_predict, pos_label=1)

    np.save(str(name) + "_fpr", fpr)
    np.save(str(name) + "_tpr", tpr)

    roc_auc = metrics.auc(fpr, tpr)

    print('AUC: ' + str(roc_auc))

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(str(name) + ".pdf")
    plt.close()


def plot(rankerValues, real_y, tuple_y, name):
    # colors = (cm.rainbow(np.linspace(0, 1, len(np.unique((real_y))))))
    colors = [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.4660, 0.6740, 0.1880]]
    dict_of_idx = defaultdict(list)
    dict_of_y_values = defaultdict(list)

    for idx, y in enumerate(tuple_y):
        dict_of_y_values[y[0]].append(rankerValues[idx])
        dict_of_y_values[y[1]].append(rankerValues[idx])
        dict_of_idx[y[0]].append(idx)
        dict_of_idx[y[1]].append(idx + 1)

    for idx, k in enumerate(sorted(dict_of_y_values.keys())):
        plt.scatter(dict_of_idx[k], dict_of_y_values[k], color=colors[int(idx)])  # , edgecolors='black')
    plt.legend(np.unique(real_y), loc=1, title="Class labels")

    plt.xlabel(r'$n$')
    plt.ylabel(r'$r(d_n,d_{n+1})$')
    plt.grid(True)
    plt.savefig(name + ".pdf")
    plt.close()


def auc_cls(estimator, X, y, w, cnn_bdt=False, linear=False, reorder=True, use_weights=True):
    if cnn_bdt:
        prediction = estimator.predict_proba(X)[:, 1]
    else:
        if linear:
            prediction = estimator.predict(X)
        else:
            prediction = estimator.predict_proba(X)
    if use_weights:
        fpr, tpr, thresholds = metrics.roc_curve(y, prediction, sample_weight=w, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y, prediction, pos_label=1)
    order = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[order], tpr[order]
    # print("AUC: " + str(metrics.auc(fpr, tpr)))
    return metrics.auc(fpr, tpr)


def auc_value(prediction, y, w, reorder=True):
    fpr, tpr, thresholds = metrics.roc_curve(y, prediction, sample_weight=w, pos_label=1)
    return metrics.auc(fpr, tpr, reorder=reorder)


def s_0_cls(estimator, X, y, w, cnn_bdt=False, linear=False, reorder=True, return_th=False, use_weights=True, b_r=0):
    if cnn_bdt:
        prediction = estimator.predict_proba(X)[:, 1]
    else:
        if linear:
            prediction = estimator.predict(X)
        else:
            prediction = estimator.predict_proba(X)
    if use_weights:
        fpr, tpr, thresholds = metrics.roc_curve(y, prediction, sample_weight=w, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y, prediction, pos_label=1)
    if b_r == 0:
        tpr = tpr[fpr > 0]
        fpr = fpr[fpr > 0]

    if reorder:
        order = np.lexsort((tpr, fpr))
        fpr, tpr = fpr[order], tpr[order]

    s0 = np.sqrt(2 * ((tpr + fpr + b_r) * np.log((1 + tpr / (fpr + b_r))) - tpr))

    s0_max = s0.max()

    if return_th:
        s0_max_th = thresholds[s0.argmax()]
        pred = np.array(estimator.predict(X, threshold=s_0_max_th))
        counts = np.unique(y[pred == 1], return_counts=True)
        return s0_max, s0_max_th, counts
    else:
        return s0_max


def prc_cls(estimator, X, y, cnn_bdt):
    if cnn_bdt:
        prediction = estimator.predict_proba(X)[:, 1]
    else:
        prediction = estimator.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, prediction, pos_label=1)
    print("PRC: " + str(metrics.auc(recall, precision)))
    return metrics.auc(recall, precision)


def nDCG_cls(estimator, X, y, at=10, cnn_bdt=False):
    if cnn_bdt:
        prediction = estimator.predict_proba(X)[:, 1]
    else:
        prediction = estimator.predict_proba(X)

    rand = np.random.random(len(prediction))
    sorted_list = [yi for _, _, yi in sorted(zip(prediction, rand, y), reverse=True)]
    yref = sorted(y, reverse=True)

    DCG = 0.
    IDCG = 0.
    for i in range(at):
        DCG += (2 ** sorted_list[i] - 1) / np.log2(i + 2)
        IDCG += (2 ** yref[i] - 1) / np.log2(i + 2)
    nDCG = DCG / IDCG
    return nDCG


def plot_sorted_list(estimator, X, y, name):
    # FIXME quite ugly but I don"t see a better way for doing this inside the sklearn gridsearch framwork ....

    X = np.array(X)
    y = np.array(y)

    data = np.concatenate((X, y), axis=1)
    compar = partial(comparator, estimator=estimator)
    if sys.version_info.major == 3:
        data = np.array(sorted(data, key=cmp_to_key(compar), reverse=True))
    else:
        data = np.array(sorted(data, cmp=compar, reverse=True))

    plot_values = []
    real_y = []
    tuple_y = []
    for idx, value in enumerate(data):
        if idx == len(data) - 1:
            break
        res = estimator.evaluate(data[idx][:-1], data[idx + 1][:-1])
        plot_values.append(res[0][0])
        tuple_y.append((data[idx][-1], data[idx + 1][-1]))
        real_y.append(int(data[idx][-1]))
        real_y.append(int(data[idx + 1][-1]))

    np.save("plot_values", plot_values)
    np.save("real_y", real_y)
    np.save("tuple_y", tuple_y)

    plot(plot_values[:100], real_y[:100], tuple_y[:100], name)


def create_roc(real_class, event_weight):
    """
    Creates a ROC and returns arrays of TP, FP and the area under curve
    features: (m,n)-dim. array (m: number of features, n: number of
    training samples) with test samples real_class: (1,n)-dim. array
    containing the real classes corresponding to 'features'
    event_weight: (1,n)-dim. array containing the weights of the
    monte-carlo simulations

    :param real_class: features
    :type real_class: array
    :param event_weight: event_weight
    :type event_weight: array
    :return: TP, FP, AUC
    :rtype: numpy array
    """
    counter = 0
    TP, FP = [0.], [0.]
    for c, w in zip(real_class, event_weight):
        if abs(c - 1) < 1e-3:
            TP.append(TP[-1] + w)
            FP.append(FP[-1])
        else:
            TP.append(TP[-1])
            FP.append(FP[-1] + w)
        counter += 1

    TP.append(1. if TP[-1] == 0 else TP[-1])
    FP.append(1. if FP[-1] == 0 else FP[-1])
    TP, FP = np.array(TP), np.array(FP)
    TP = TP / TP[-1] * 100
    FP = FP / FP[-1] * 100

    # Calculate AUC
    AUC = 0.
    for i in range(len(TP) - 1):
        AUC += TP[i] * (FP[i + 1] - FP[i])
    AUC /= 10000

    return TP, FP, AUC


def seperationPlot(dr, x_train, x_test, y_train, y_test):
    nn0output_train = getNetOutput(dr, x_train)
    nn0output_test = getNetOutput(dr, x_test)

    sig_train = np.array(nn0output_train[np.where(y_train == 1)])
    bgk_train = np.array(nn0output_train[np.where(y_train == 0)])

    sig_test = np.array(nn0output_test[np.where(y_test == 1)])
    bgk_test = np.array(nn0output_test[np.where(y_test == 0)])

    low = min(np.min(d) for d in [sig_train, bgk_train, sig_test, bgk_test])
    high = max(np.max(d) for d in [sig_train, bgk_train, sig_test, bgk_test])
    low_high = (low, high)

    plt.hist(sig_train, 30, label='higgs (train)', color="r", alpha=0.5, histtype='stepfilled', range=low_high,
             normed=True)  # density=True)
    plt.hist(bgk_train, 30, label='ttbar (train)', color="b", alpha=0.5, histtype='stepfilled', range=low_high,
             normed=True)  # density=True

    hist, bins = np.histogram(sig_test, bins=30, range=low_high, normed=True)  # density=True)
    scale = len(sig_test) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    # width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='higgs (test)')

    hist, bins = np.histogram(bgk_test, bins=30, range=low_high, normed=True)  # density=True)
    scale = len(bgk_test) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    # width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='ttbar (test)')

    plt.xlabel("ranker output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig('separation_plots_ranker.pdf')
    plt.close()


def aucScorer(estimator, X, y):
    """
    ToDo: remove n=100??
    You can generate even more flexible model scorers by constructing your own scoring object from scratch, without using the make_scorer factory. For a callable to be a scorer, it needs to meet the protocol specified by the following two rules:

    It can be called with parameters (estimator, X, y), where estimator is the model that should be evaluated, X is validation data, and y is the ground truth target for X (in the supervised case) or None (in the unsupervised case).
    It returns a floating point number that quantifies the estimator prediction quality on X, with reference to y. Again, by convention higher numbers are better, so if your scorer returns loss, that value should be negated.
    """
    data = np.concatenate((X, y), axis=1)
    compar = partial(comparator, estimator=estimator)
    if sys.version_info.major == 3:
        data = np.array(sorted(data, key=cmp_to_key(compar), reverse=True))
    else:
        data = np.array(sorted(data, cmp=compar, reverse=True))

    AUC = create_roc(data[:, -1], np.ones(len(data[:, -1])))
    return AUC


def readClassiData(data_path=None, number_features=13, preprocessing="quantile_3", skip_head=False, target_first=True):
    x = []
    y = []
    for line in open(data_path):
        if skip_head:
            skip_head = False
            continue
        s = line.split(",")
        if target_first:
            y.append(int(s[0]))
            k = 1
        else:
            y.append(int(s[-1]))
            k = 0
        x.append(np.zeros(number_features))
        for i in range(number_features):
            x[-1][i] = float(s[i + k])
    if preprocessing == "quantile_3":
        x = QuantileTransformer(output_distribution="normal").fit_transform(x) / 3
    else:
        x = np.array(x)
    return np.array(x), np.array([y]).transpose()


def readData(data_path=None, debug_data=False, binary=False, preprocessing="quantile_3", at=10, number_features=136):
    """
    Function for reading the letor data
    :param binary: boolean if the labels of the data should be binary
    :param preprocessing: if set QuantileTransformer(output_distribution="normal") is used
    :param at: if the number of documents in the query is less than "at" they will not be
               taken into account. This is needed for calculating the ndcg@k until k=at.
    :return: list of queries, list of labels and list of query id
    """
    if debug_data:
        path = "test_data.txt"
    elif data_path is not None:
        path = data_path

    x = []
    y = []
    q = []
    for line in open(path):
        s = line.split()
        if binary:
            if int(s[0]) > 1.5:
                y.append(1)
            else:
                y.append(0)
        else:
            y.append(int(s[0]))

        q.append(int(s[1].split(":")[1]))

        x.append(np.zeros(number_features))
        for i in range(number_features):
            x[-1][i] = float(s[i + 2].split(":")[1])

    if preprocessing == "quantile_3":
        x = QuantileTransformer(
            output_distribution="normal").fit_transform(x) / 3
    else:
        x = np.array(x)
    y = np.array([y]).transpose()
    q = np.array(q)
    xt = []
    yt = []

    for qid in np.unique(q):
        cs = []
        if len(y[q == qid]) < at:
            continue
        for yy in y[q == qid][:, 0]:
            if yy not in cs:
                cs.append(yy)
        if len(cs) == 1:
            continue
        xt.append(x[q == qid])
        yt.append(y[q == qid])

    return np.array(xt), np.array(yt), q


def comparator(x1, x2, estimator):
    """
    :param x1: list of documents
    :param x2: list of documents
    :param estimator: class of directRanker
    :return: cmp value for sorting the query
    """
    res = estimator.evaluate(x1[:-1], x2[:-1])
    if res < 0:
        return -1
    elif res > 0:
        return 1
    return 0


def getNetOutput(estimator, X):
    """
    :param x1: list of documents
    :param x2: list of documents
    :param estimator: class of directRanker
    :return: cmp value for sorting the query
    """
    res = estimator.evaluatePartNet(X)
    return res


def nDCGScorer(estimator, X, y, at=20):
    """
    :param estimator: class of directRanker
    :param X: list of queries
    :param y: list of label per queries
    :param at: value until the nDCG should be calculated
    :return: cmp value for sorting the query
    """
    compare = partial(comparator, estimator=estimator)

    listOfnDCG = []
    for query, y_query in zip(X, y):
        DCG = 0.
        IDCG = 0.

        data = np.concatenate((query, y_query), axis=1)
        if sys.version_info.major == 3:
            data = np.array(
                sorted(data, key=cmp_to_key(compare), reverse=True))
        else:
            data = np.array(sorted(data, cmp=compare, reverse=True))
        yref = np.array(sorted(y_query[:, 0], reverse=True))
        for i in range(at):
            DCG += (2 ** data[i, -1] - 1) / np.log2(i + 2)
            IDCG += (2 ** yref[i] - 1) / np.log2(i + 2)
        listOfnDCG.append(DCG / IDCG)
    nDCG = np.mean(listOfnDCG)
    print("nDCG@" + str(at) + ": " + str(round(nDCG, 4)) + " +- " + str(round(np.std(listOfnDCG), 4)))
    return nDCG


def AvgP_cls(estimator, X, y):
    avgp = average_precision_score(y, estimator.predict_proba(X))
    print('AvgP: ' + str(avgp))
    return avgp


def MAP_cls(estimator, X, y):
    listOfAvgP = []
    for query, y_query in zip(X, y):
        listOfAvgP.append(AvgP_cls(estimator, query, y_query))

    print(np.mean(listOfAvgP), np.std(listOfAvgP))
    return np.mean(listOfAvgP)
