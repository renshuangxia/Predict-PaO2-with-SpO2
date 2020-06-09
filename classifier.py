import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection,  linear_model,  metrics, svm, base

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import itemfreq
from scipy import interp

import matplotlib.pyplot as plt
from joblib import dump, load

import argparse, os
from RegscorePy import bic

np.random.seed(1)

def read_data(path, multi_class=False, seven_features=False, fitlerSpO2=True):
    print("Reading data from",path)
    df = pd.read_csv(path)

    # delete example with Sao2 value larger than 800
    df = df.drop(df[df['Sao2']>800].index)

    # convert temperature to correct form
    # print("before_covert_temp",sum(df['Temperature']>50))
    # print("before_covert_fio2",sum(df['Fio2']>1))
    for i in range(0,df.shape[0]):
        if df.iloc[i,9] >50:
            df.iloc[i,9] = (df.iloc[i,9]-32)/1.8
        if df.iloc[i,5] >1:
            df.iloc[i, 5] = df.iloc[i, 5] / 100
    # print("after_covert_fio2",sum(df['Fio2']>1))

    df = df[pd.notnull(df['Fio2'])]
    df = df [pd.notnull(df['Pao2'])]

    # delete sample which has Fio2=0
    df = df[df.Fio2!=0]
    # delete sample which has Spo2<60
    df = df[60<=df.Spo2]
    # delete sample which has Peep>=40
    df = df[40>=df.Peep]

    # delete spo2 > 96
    if fitlerSpO2:
        df = df[97 > df.Spo2]

    df.info()

    # plt.scatter(df['Spo2'],df['Pao2'])
    # plt.show()

    Spo2_Fio2_log = np.log10(df['Spo2']/df['Fio2'])
    Pao2_Fio2_log = np.log10(df['Pao2']/df['Fio2'])
    # plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,s=2)
    # plt.xlabel('log(S/F)')
    # plt.ylabel('log(P/F)')
    # plt.show()

    print("female ratio",1-sum(df['gender'])/len(df['gender']),len(df['gender']))
    print("age", np.mean(df['age']),np.std(df['age']))
    hist = df.hist()
    plt.show()

    if seven_features:
        df = df[pd.notnull(df['Map'])]
        df = df[pd.notnull(df['Temperature'])]
        #delete sample which has Vt>=4000
        df = df[4000>=df.Vt]
        data = np.array([df['Spo2'],df['Fio2'],df['Peep'],df['Vt'],df['Map'],df['Temperature'],df['vaso'], df['Pao2']/df['Fio2']])
    else:
        data = np.array([df['Spo2'], df['Fio2'], df['Peep'], df['Pao2']/df['Fio2']])
    data = data.T
    print(data.shape,data[:,-1])

    # subjectid_unique_value_freq = itemfreq()
    nodes, inv, counts = np.unique(df['subject_id'], return_inverse=True, return_counts=True)
    subjectid_unique_value_freq = itemfreq(counts)
    print('counts', counts, 'subjectid_unique_value_freq', subjectid_unique_value_freq)

    if multi_class: # for multi-class learning
        # convert y_test into category
        list3, list2, list1, list0 = [], [], [], []
        for i in range(0, data.shape[0]):
            if data[i, -1] > 300:
                list3.append(data[i,-1])
                data[i, -1] = 3
            elif 200 <data[i,-1] and data[i,-1]<=300:
                list2.append(data[i, -1])
                data[i, -1] = 2
            elif 100 <data[i,-1]and data[i,-1]<=200:
                list1.append(data[i, -1])
                data[i, -1] = 1
            else:
                list0.append(data[i, -1])
                data[i, -1] = 0
        print('>300', np.mean(list3), np.std(list3), np.shape(list3))
        print('200-300', np.mean(list2), np.std(list2), np.shape(list2))
        print('100-200', np.mean(list1), np.std(list1), np.shape(list1))
        print('>100', np.mean(list0), np.std(list0), np.shape(list0))
        print("pf unique value number", itemfreq(data[:, -1]))
        print('data sum1:', sum(data[:, -1]))
    else:
        for i in range(0, data.shape[0]):
            data[i, -1] = 0 if data[i, -1] <= 150 else 1

    return data, df


def normalize(X_train, X_test, seven_features=False):
    input_scaler = StandardScaler()

    if seven_features:
        input_scaler.fit(X_train[:, 0:-1])
        X_train_scale = input_scaler.transform(X_train[:, 0:-1])
        X_test_scale = input_scaler.transform(X_test[:, 0:-1])
        X_train_scale = np.concatenate((X_train_scale, X_train[:,-1].reshape(len(X_train[:,-1]),1)),axis = 1)
        X_test_scale = np.concatenate((X_test_scale, X_test[:,-1].reshape(len(X_test[:,-1]),1)),axis = 1)
    else:
        input_scaler.fit(X_train)
        X_train_scale = input_scaler.transform(X_train)
        X_test_scale = input_scaler.transform(X_test)
    return X_train_scale, X_test_scale

# define accuracy
def accuracy(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    PPV = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    NPV = tn / (fn + tn)
    pos_lr = sensitivity / (1 - specificity)
    neg_lr = (1 - sensitivity) / specificity
    recall = tp / (tp + fn)
    precison = tp / (tp + fp)
    f1 = (2 * precison * recall)/(precison + recall)
    print("*F1: %f  *Sensitivity: %f  *Specificity: %f  *PPV: %f  *NPV: %f  *Positive-LR: %f  *Negative-LR: %f" %
          (f1, sensitivity, specificity, PPV, NPV, pos_lr, neg_lr))
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

# other two methods
def log_linear(S,F):
    P = 0.48 + 0.78 * np.log10(S/F)
    Pao2 = F * np.power(10,P)
    return Pao2

def non_linear(S):
    S = S/100
    K = 1/S-1
    I = 11700/K
    M = np.power(50,3)+np.power(I,2)
    Plus = I + np.power(M,1./2)
    Minus = I - np.power(M,1./2)
    Part1 =  np.cbrt(Plus)
    Part2 =  np.cbrt(Minus)
    Pao2 = Part1 + Part2
    return Pao2



def conv_cate(column):
    '''
    for row in range(0,len(column)):
        if column[row] >300:
            column[row] = 3
        elif  200<column[row] <=300:
            column[row] = 2
        elif 100<column[row] <=200:
            column[row] = 2
        else:
            column[row] = 1
    '''
    for row in range(0,len(column)):
        column[row] = 0 if column[row] <= 150 else 1
    return column


def get_baseline(X_test, y_test):
    test_spo2 = X_test[:,0]
    test_fio2 = X_test[:,1]
    real_class = y_test

    print("test_fio2",test_fio2)
    print("realclass",real_class)

    pred_pao2_loglinear = log_linear(test_spo2,test_fio2)

    pred_pfratio_loglinear = pred_pao2_loglinear/test_fio2
    pred_class_loglinear = conv_cate(pred_pfratio_loglinear)
    cm_loglinear = confusion_matrix(real_class,pred_class_loglinear)
    print("F1 logLinear: ", metrics.f1_score(real_class, pred_pfratio_loglinear))
    print("print loglinear size: ", real_class.shape)
    #print(cm_loglinear)
    #print('pred_class_loglinear:', pred_class_loglinear)
    print("accuracy_loglinear",accuracy(cm_loglinear))
    print("loglinear accuracy: ", metrics.accuracy_score(real_class, pred_pfratio_loglinear))
    print('\n')
    test_spo2[test_spo2>99.9] = 99.6
    pred_pao2_nonlinear = non_linear(test_spo2)
    pred_pfratio_nonlinear = pred_pao2_nonlinear/test_fio2
    pred_class_nonlinear = conv_cate(pred_pfratio_nonlinear)
    print("F1 nonLinear: ", metrics.f1_score(real_class, pred_class_nonlinear))
    print("print nonlinear size: ", real_class.shape)
    cm_nonlinear = confusion_matrix(real_class,pred_class_nonlinear)
    #print(cm_nonlinear)
    print("accuracy_nonlinear",accuracy(cm_nonlinear))
    print("nonlinear accuracy: ", metrics.accuracy_score(real_class, pred_class_nonlinear))
    unique, counts = np.unique(y_test, return_counts=True)
    print('Class number: ', dict(zip(unique, counts)))

def run_classifier(path, model='logistic_regression', seven_features=False, filterSpO2=True, load_model=False):
    if model == 'multi_class':
        data, df = read_data(path, multi_class=True, seven_features=seven_features, fitlerSpO2=filterSpO2)
    else:
        data, df = read_data(path, seven_features=seven_features, fitlerSpO2=filterSpO2)

    if filterSpO2:
        subSpO2Folder = 'FilteredSpO2/'
    else:
        subSpO2Folder = 'NoSpO2Filtering/'

    if seven_features:
        n_features = 7
    else:
        n_features = 3

    print("-----------------------------------------------")
    if model == 'multi_class':
        print("        MULTI CLASS CLASSIFIER")
    elif model == 'svc':
        print("        SUPPORT VECTOR CLASSIFIER")
    elif model == 'mlp':
        print("   MULTI-LAYER PERCEPTRON")
    else:
        print("             LOGISTIC REGRESSION")
    print("-----------------------------------------------")


    X = data[:, 0:-1]
    y = data[:, -1]
    n_split = 10
    kFold = model_selection.KFold(n_splits=n_split, shuffle=True, random_state=1)
    f1_scores = []
    bic_scores = []
    i = 0
    res = np.zeros((8, n_split))
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in kFold.split(X, y):
        print('Fold:',i + 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scale, X_test_scale = normalize(X_train, X_test, seven_features=seven_features)
        print('x train:',X_train_scale.shape)
        if model == 'svc':
            model_name = 'SVC'
            classifier = svm.SVC(gamma='auto', kernel='rbf', probability=True)
        elif model == 'sgd':
            model_name = 'SGD Classifier'
            classifier = linear_model.SGDClassifier(loss='log')
        elif model == 'ridge':
            model_name = 'Ridge Classifier'
            classifier = linear_model.RidgeClassifier()
        elif model == 'mlp':
            model_name = 'Neural Network'
            if seven_features:
                classifier = MLPClassifier((12, 8, 6, 4, 4), max_iter=200, activation='tanh', solver='adam', random_state=1, momentum=0.8) # current BEST 0.7798 accuracy
            else:
                classifier = MLPClassifier((6, 3), max_iter=200, activation='tanh', solver='adam', random_state=1, momentum=0.6) # current best


        else:
            model_name = 'Logistic Regression'
            classifier = linear_model.LogisticRegression()

        final_classifier = base.clone(classifier)
        if load_model: # load a pretrained model
            classifier = load('saved_models/classifer/' + subSpO2Folder + model_name.replace(' ', '_') + '_'  + str(n_features) + '_features.joblib')
        else: # train a new model
            classifier.fit(X_train_scale, y_train)

        y_pred = classifier.predict(X_test_scale)
        y_pred_prob = classifier.predict_proba(X_test_scale)[:,1]

        f1_scores.append(metrics.f1_score(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        res[0][i] = tp / (tp + fp) # PPV
        res[1][i] = tp / (tp + fn) # sensitivity
        res[2][i] = tn / (tn + fp) # specificity
        res[3][i] = tn / (fn + tn) # NPV
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        res[4][i] = roc_auc
        res[5][i] = res[1][i]/(1 - res[2][i]) # Positive Likelyhood rato
        res[6][i] = (1 - res[1][i]) / res[2][i]  # Negative Likelyhood rato
        res[7][i] = metrics.accuracy_score(y_test, y_pred)
        bic_val = bic.bic(y_test, y_pred_prob, n_features)
        bic_scores.append(bic_val)

        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        print('current f1:', f1_scores[-1], ' PPV:', res[0][i], ' Sensitivity:', res[1][i],
              ' Specificity:', res[2][i], ' NPV:', res[3][i], ' BIC:', bic_val,
              'ROAUC:',res[4][i], 'Accuracy:',res[7][i])
        i += 1

    # train a final model using entire set
    if not load_model:
        final_classifier.fit(X, y)
        final_pred = final_classifier.predict(X)
        print('Final F1 ', metrics.f1_score(y, final_pred))
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')

        if not os.path.exists('saved_models/classifer'):
            os.mkdir('saved_models/classifer')

        model_output_path = 'saved_models/classifer/' + subSpO2Folder + model_name.replace(' ', '_') + '_' + str(n_features) + '_features.joblib'
        dump(final_classifier, model_output_path)

    res = np.mean(res, axis=1)

    print('\n=================================================')
    print('                      RESULT')
    print("--------------------------------------------------")
    print('  ', model_name)
    print('   Total cases:', y.shape)
    print('Seven Features:', seven_features, '  FilterSpO2:', filterSpO2)
    print('       Mean F1:', np.mean(np.array(f1_scores)))
    print('           PPV:', res[0])
    print('   Sensitivity:', res[1])
    print('   Specificity:', res[2])
    print('           NPV:', res[3])
    print('         ROAUC:', res[4])
    print('   Positive LR:', res[5])
    print('   Negative LR:', res[6])
    print('           BIC:', np.mean(np.array(bic_scores)))
    print('      Accuracy:', res[7])
    print("\nBaseline")
    get_baseline(np.array([df['Spo2'], df['Fio2']]).T, y)
    print('==================================================')

    if model == 'multi_class':
        return

    # Plot ROAUC
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (res[4]), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + ' AUC (' + str(n_features) + ' features)')
    plt.legend(loc="lower right")
    plt.text(0.32, 0.7, 'More accurate area', fontsize=12, color='green')
    plt.text(0.63, 0.4, 'Less accurate area', fontsize=12, color='red')
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',dest='data', type=str,
                        default='new_data.csv', help='Path to data file')
    parser.add_argument('--model', dest='model', type=str,
                        default='logistic_regression', help='model to run: logistic_regression, sgd, mlp, svc')
    parser.add_argument('--seven_features', dest='seven_features', type=bool,
                        default=True, help='If true, use 7 input features, if false, use 3 input features')
    parser.add_argument('--fitlerSpO2', dest='filterSpO2', type=bool,
                        default=False, help='Whether or not to filter samples with SpO2 value greater than 96')
    parser.add_argument('--load_model', dest='load_model', type=bool,
                       default=False, help='Whether or not to load pretrained model') # model path is hard coded
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = args.data
    model = args.model
    seven_features = args.seven_features
    fitlerSpO2 = args.filterSpO2
    load_model = args.load_model

    #run_classifier(path, model=model, seven_features=seven_features, fitlerSpO2=fitlerSpO2, load_model=load_model)

    run_classifier(path, model='mlp', seven_features=False, filterSpO2=False, load_model=False)


if __name__ == '__main__':
    main()
