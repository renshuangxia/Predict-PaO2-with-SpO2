import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection,  linear_model,  metrics, svm

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import argparse

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
    print('pred_class_loglinear:', pred_class_loglinear)
    print("accuracy_loglinear",accuracy(cm_loglinear))

    test_spo2[test_spo2>99.9] = 99.6
    pred_pao2_nonlinear = non_linear(test_spo2)
    pred_pfratio_nonlinear = pred_pao2_nonlinear/test_fio2
    pred_class_nonlinear = conv_cate(pred_pfratio_nonlinear)
    cm_nonlinear = confusion_matrix(real_class,pred_class_nonlinear)
    print("accuracy_nonlinear",accuracy(cm_nonlinear))
    unique, counts = np.unique(y_test, return_counts=True)
    print('Class number: ', dict(zip(unique, counts)))


def run_mc_classifier(path):
    # train test split
    data, _ = read_data(path, multi_class=True)
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.3, random_state=1)
    X_train_scale, X_test_scale = normalize(X_train, X_test)
    classifier = MLPClassifier((5, 10, 5), max_iter=50, activation = 'tann',solver='adam',random_state=1)
    classifier.fit(X_train_scale,y_train)
    y_pred = classifier.predict(X_test_scale)
    cm_nn = confusion_matrix(y_test, y_pred)
    print('y_test:',y_test[0:50])
    print('y_pred:', y_pred[0:50])
    print("f1:",metrics.f1_score(y_test, y_pred, average='micro')) #use micro or macro F1 instead of accuracy


def run_classifier(path, model='logistic_regression', seven_features=False, fitlerSpO2=True):
    if model == 'multi_class':
        data, df = read_data(path, multi_class=True, seven_features=seven_features, fitlerSpO2=fitlerSpO2)
    else:
        data, df = read_data(path, seven_features=seven_features, fitlerSpO2=fitlerSpO2)

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
    stratifier = model_selection.StratifiedKFold(n_splits=10, shuffle=False)
    f1_scores = []
    i = 1
    for train_index, test_index in stratifier.split(X, y):
        print('Fold:',i)
        i+=1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scale, X_test_scale = normalize(X_train, X_test, seven_features=seven_features)
        print('x train:',X_train_scale.shape)
        if model == 'multi_class':
            classifier = MLPClassifier((5, 10, 5), max_iter=200, activation='tanh', solver='adam', random_state=1)
        elif model == 'svc':
            classifier = svm.SVC(gamma='auto', kernel='rbf')
        elif model == 'sgd':
            classifier = linear_model.SGDClassifier(loss='log')
        elif model == 'mlp':
            if seven_features:
                classifier = MLPClassifier((16, 8, 4), max_iter=50, activation='tanh', solver='adam', random_state=1)
            else:
                classifier = MLPClassifier((6, 3), max_iter=50, activation='tanh', solver='adam', random_state=1)
        else:
            classifier = linear_model.LogisticRegression()
        classifier.fit(X_train_scale, y_train)
        y_pred = classifier.predict(X_test_scale)

        if model == 'multi_class':
            f1_scores.append(metrics.f1_score(y_test, y_pred, average='micro'))
        else:
            f1_scores.append(metrics.f1_score(y_test, y_pred))
            print('curren f1:', f1_scores[-1])


    print('\n=================================================')
    print('                      RESULT')
    print("--------------------------------------------------")
    print(model + " Mean F1:", np.mean(np.array(f1_scores)))
    print("Baseline")
    get_baseline(np.array([df['Spo2'], df['Fio2']]).T, y)
    print('==================================================')


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
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = args.data
    model = args.model
    seven_features = args.seven_features
    fitlerSpO2 = args.filterSpO2
    run_classifier(path, model=model, seven_features=seven_features, fitlerSpO2=fitlerSpO2)


if __name__ == '__main__':
    main()
