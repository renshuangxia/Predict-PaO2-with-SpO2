import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, clone_model, load_model, save_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras import backend
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection,  linear_model, svm, base
from sklearn.metrics import mean_squared_error
from RegscorePy import bic
from tensorflow import set_random_seed
from joblib import dump, load
import argparse, os
import warnings

warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(1)
# set_random_seed(1)
tf.compat.v1.set_random_seed

def init_data(path='new_data.csv', seven_features=False, fitlerSpO2=False):
    df = pd.read_csv(path)

    # delete example which Sao2 value larger than 800
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
    if seven_features:
        df = df[pd.notnull(df['Map'])]
        df = df[pd.notnull(df['Temperature'])]
        #delete sample which has Vt>=4000
        df = df[4000>=df.Vt]
        data = np.array([df['Spo2'], df['Fio2'], df['Peep'], df['Vt'], df['Map'], df['Temperature'], df['vaso'], df['Pao2']])
    else:
        data = np.array([df['Spo2'], df['Fio2'], df['Peep'], df['Pao2']])
    data = data.T

    # R square between SF ratio and PF ratio
    PF_ratio = data[:, -1] / data[:, 1]
    SF_ratio = data[:, 0] / data[:, 1]
    _, _, r_value, _, _ = stats.linregress(np.log10(SF_ratio), np.log10(PF_ratio))
    r_square = r_value ** 2
    print('R value:', r_value)
    print('R Square between SF ratio and PF ratio: ', r_square)
    return data

def plot(df):
    # plt.scatter(df['Spo2'],df['Pao2'])
    # plt.show()
    Spo2_Fio2_log = np.log10(df['Spo2']/df['Fio2'])
    Pao2_Fio2_log = np.log10(df['Pao2']/df['Fio2'])

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,s=2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    plt.savefig('log_sf_pf.png')
    plt.show()

    plt.scatter(df['Pao2'],df['Sao2'],s = 2)
    plt.xlabel('Pao2')
    plt.ylabel('Sao2')
    plt.savefig('Pao2_sao2.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['Vt'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('Vt')
    plt.savefig('log_sf_pf_vt.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['Peep'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('Peep')
    plt.savefig('log_sf_pf_peep.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['Map'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('Map')
    plt.savefig('log_sf_pf_map.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['Temperature'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('Temperature')
    plt.savefig('log_sf_pf_temp.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['vaso'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('vaso')
    plt.savefig('log_sf_pf_vaso.png')
    plt.show()

    plt.scatter(Spo2_Fio2_log,Pao2_Fio2_log,c = df['Paco2'],s = 2)
    plt.xlabel('log(S/F)')
    plt.ylabel('log(P/F)')
    cbar = plt.colorbar()
    cbar.set_label('Paco2')
    plt.savefig('log_sf_pf_paco2.png')
    plt.show()

    plt.scatter(df['Spo2'],df['Sao2'])
    plt.xlabel('Spo2')
    plt.ylabel('Sao2')
    plt.savefig('spo2_sao2.png')
    plt.show()

    hist = df.hist()
    plt.show()


#calculate rmse
def rmse(y_true,y_predict):
    return backend.sqrt(backend.mean(backend.square(y_predict-y_true),axis=-1))


def normalize(X_train, X_test, y_train, y_test, seven_features=False):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    print('***train size: ', X_train.shape)
    if seven_features:
        input_scaler.fit(X_train[:, 0:-1])
        X_train_scale = input_scaler.transform(X_train[:, 0:-1])
        X_test_scale = input_scaler.transform(X_test[:, 0:-1])
        X_train_scale = np.concatenate((X_train_scale,X_train[:,-1].reshape(len(X_train[:,-1]),1)),axis = 1)
        X_test_scale = np.concatenate((X_test_scale,X_test[:,-1].reshape(len(X_test[:,-1]),1)),axis = 1)
    else:
        input_scaler.fit(X_train)
        X_train_scale = input_scaler.transform(X_train)
        X_test_scale = input_scaler.transform(X_test)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    output_scaler.fit(y_train)
    y_train_scale = output_scaler.transform(y_train)
    y_test_scale = output_scaler.transform(y_test)
    return X_train_scale, X_test_scale,y_train_scale, y_test_scale, output_scaler

def normalize_X_y(X, y, seven_features=False):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    if seven_features:
        input_scaler.fit(X[:, 0:-1])
        X_scale = input_scaler.transform(X[:, 0:-1])
        X_scale = np.concatenate((X_scale, X[:,-1].reshape(len(X[:,-1]),1)),axis = 1)
    else:
        input_scaler.fit(X)
        X_scale = input_scaler.transform(X)

    y = y.reshape(len(y), 1)
    output_scaler.fit(y)
    y_scale = output_scaler.transform(y)
    return X_scale, y_scale, output_scaler

def np_rmse(arr1,arr2):
    return np.sqrt(np.mean((arr1-arr2)**2))


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

# compute baseline results
def get_baseline(X, y):
    test_spo2 = X[:,0]
    test_fio2 = X[:,1]
    test_pao2 = y
    pfratio = test_pao2/test_fio2
    pred_pao2_loglinear = log_linear(test_spo2,test_fio2)
    loglinear_rmse = np_rmse(y,pred_pao2_loglinear)
    pred_pfratio_loglinear = pred_pao2_loglinear/test_fio2
    #print("pfratio",pfratio,"pf_log",pred_pfratio_loglinear)

    #print("loglinear_rmse", loglinear_rmse, "loglinear_pred_y", pred_pao2_loglinear, "y_test_loglinear", y)

    test_spo2[test_spo2>99.9] = 99.6
    pred_pao2_nonlinear = non_linear(test_spo2)
    nonlinear_rmse = np_rmse(y,pred_pao2_nonlinear)
    pred_pfratio_nonlinear = pred_pao2_nonlinear/test_fio2
    #print("nonlinear_rmse", nonlinear_rmse, "nonlinear_pred_y", pred_pao2_nonlinear, "y_test_nonlinear", y)

    return loglinear_rmse, nonlinear_rmse, pfratio, pred_pfratio_loglinear,pred_pfratio_nonlinear


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')


# use neural network, no n-fold verion, test size 30%
# Not used in function run_predictor
def nn_predictor(data):
    X_train, X_test, y_train, y_test  = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.3, random_state=1)
    X_train_scale, X_test_scale, y_train_scale, y_test_scale, output_scaler = normalize(X_train, X_test, y_train, y_test)


    # model configuration and initialization
    model = Sequential()
    model.add(Dense(16, input_dim=3, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

    history = model.fit(X_train_scale, y_train_scale, epochs=50, batch_size=50,validation_data=(X_test_scale,y_test_scale))
    test_pred = model.predict(X_test_scale)
    print('weights',model.get_weights())

    inverse_test_pred = output_scaler.inverse_transform(test_pred)
    inverse_test = output_scaler.inverse_transform(y_test_scale)
    pred_pao2_nn = np.array(inverse_test_pred).flatten()
    rmse_result = np_rmse(np.array(inverse_test_pred).flatten(), np.array(y_test).flatten())

    print("nn_rmse", rmse_result, "nn_pred_y", np.array(inverse_test_pred).flatten(), "y_test_nn", np.array(y_test).flatten())
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    loglinear_rmse, nonlinear_rmse, pfratio, pred_pfratio_loglinear, pred_pfratio_nonlinear = get_baseline(X_test, y_test)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    test_fio2 = X_test[:, 1]
    pred_pfratio_nn = pred_pao2_nn / test_fio2

    ax1.scatter(pfratio, pfratio - pred_pfratio_nonlinear, s=2, c='r', marker="o", label='nonlinear')
    ax1.scatter(pfratio, pfratio - pred_pfratio_loglinear, s=2, c='b', marker="s", label='loglinear')
    ax1.scatter(pfratio, pfratio - pred_pfratio_nn, s=2, c='y', marker="*", label='neural network')

    plt.xlabel('Measured PF')
    plt.ylabel('Measured PF - Imputed PF')
    plt.legend(loc='upper left')
    plt.savefig('measured pf and imputed pf.png')
    plt.show()
    plt.close()


def run_predictor(path, model='linear_regression', seven_features=False, filterSpO2=False, load_model=False):
    data = init_data(path, seven_features=seven_features, fitlerSpO2=filterSpO2)
    X = data[:, 0:-1]
    y = data[:, -1]

    if filterSpO2:
        subSpO2Folder = 'FilteredSpO2/'
    else:
        subSpO2Folder = 'NoSpO2Filtering/'

    if seven_features:
        n_features = 7
    else:
        n_features = 3

    kFold = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
    rmse_results = []
    bic_scores = []
    total_inversed_preds = []
    i = 1
    for train_index, test_index in kFold.split(X, y):
        print("Current fold:",i)
        i += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_scale, X_test_scale, y_train_scale, y_test_scale, output_scaler = normalize(X_train, X_test, y_train, y_test, seven_features=seven_features)

        y_train_scale = np.array(y_train_scale).squeeze()
        y_test_scale = np.array(y_test_scale).squeeze()

        if model == 'svr':
            model_name = 'svr'
            predictor = svm.SVR(kernel='rbf')
        elif model == 'neural_network':
            model_name = 'Neural Network'
            predictor = Sequential()
            if seven_features:
                predictor.add(Dense(16, input_dim=7, activation='tanh'))
                predictor.add(Dense(8, activation='tanh'))
                predictor.add(Dense(5, activation='tanh'))
                predictor.add(Dense(1, activation='linear'))
            else:
                predictor.add(Dense(6, input_dim=3, activation='tanh'))
                predictor.add(Dense(3, activation='tanh'))
                predictor.add(Dense(1, activation='linear'))
            predictor.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
            final_predictor = clone_model(predictor)
        elif model == 'sgd':
            model_name = 'SGDRegressor'
            predictor = linear_model.SGDRegressor()
        elif model == 'ridge':
            model_name = 'Ridge'
            predictor = linear_model.Ridge()
        else:
            model_name = 'Linear Regression'
            predictor = linear_model.LinearRegression()

        if not model == 'neural_network':
            final_predictor = base.clone(predictor)

        if load_model:
            if model == 'neural_network':
                predictor = load_model('saved_models/regressor/' + subSpO2Folder + model_name.replace(' ', '_') + '_' + str(n_features) + '_features.ckpt')
            else:
                predictor = load('saved_models/regressor/' + subSpO2Folder + model_name.replace(' ', '_') + '_' + str(n_features) + '_features.joblib')
        else:
            print('start training...')
            if model == 'neural_network':
                predictor.fit(X_train_scale, y_train_scale, epochs=100, batch_size=50)
            else:
                predictor.fit(X_train_scale, y_train_scale)

        test_pred = predictor.predict(X_test_scale)

        inverse_test_pred = output_scaler.inverse_transform(test_pred)
        total_inversed_preds.append(inverse_test_pred)

        rmse_result = np_rmse(np.array(inverse_test_pred).flatten(), np.array(y_test).flatten())
        rmse_results.append(rmse_result)

        bic_val = y_test.shape[0] * np.log(rmse_result ** 2) + n_features * np.log(y_test.shape[0])
        bic_scores.append(bic_val)

        # bic_val_2 = y_test.shape[0]  * np.log(rmse_result **2) + n_features * np.log(y_test.shape[0]) # for BIC calculation validation
        # print('BIC 2: ', bic_val_2)
        print(model + ' rmse:', rmse_result, '  BIC:', bic_val)

    # train a final model using entire set
    if not load_model:
        X_scale, y_scale, _ = normalize_X_y(X, y, seven_features=seven_features)
        y_scale = np.array(y_scale).squeeze()
        if model == 'neural_network':
            final_predictor.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
            final_predictor.fit(X_scale, y_scale, epochs=100, batch_size=50)
        else:
            final_predictor.fit(X_scale, y_scale)

        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')

        if not os.path.exists('saved_models/regressor'):
            os.mkdir('saved_models/regressor')

        if model == 'neural network':
            save_model(final_predictor, 'saved_models/regressor/' + subSpO2Folder + model_name.replace(' ', '_') + '_' + str(n_features) + '_features.ckpt')
        else:
            dump(final_predictor, 'saved_models/regressor/' + subSpO2Folder + model_name.replace(' ', '_') + '_' + str(n_features) + '_features.joblib')

    loglinear_rmse, nonlinear_rmse, pfratio, pred_pfratio_loglinear, pred_pfratio_nonlinear = get_baseline(X, y)
    print('\n========================')
    print('         RESULTS')
    print('------------------------')
    print('Seven Features: ', seven_features)
    print('Filtered SpO2: ', filterSpO2)
    print('total cases:', X.shape[0])
    print("loglinear_rmse,", loglinear_rmse)
    print("nonlinear_rmse", nonlinear_rmse)
    mean_rmse = np.mean(np.array(rmse_results))
    mean_bic = np.mean(np.array(bic_scores))
    print('   ' + model)
    print('  * mean rmse:', mean_rmse)
    print('  *  mean BIC:', mean_bic)
    print('========================')
    total_inversed_preds = np.array(total_inversed_preds)

    if filterSpO2 and model == 'neural_network':
        # plot
        print("plot results...")
        pred_pao2_nn = total_inversed_preds.flatten()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        test_fio2 = X[:, 1]
        pred_pfratio_nn = pred_pao2_nn / test_fio2

        #pfratio = pfratio[np.where(pfratio <= 400)]
        plot_indicies = pfratio <= 400
        pfratio = pfratio[plot_indicies]
        diff_nonlinear = pfratio - pred_pfratio_nonlinear[plot_indicies]
        diff_loglinear = pfratio - pred_pfratio_loglinear[plot_indicies]
        diff_nn = pfratio - pred_pfratio_nn[plot_indicies]

        md = np.mean(np.concatenate([diff_nonlinear, diff_loglinear, diff_nn]))
        sd = np.std(np.concatenate([diff_nonlinear, diff_loglinear, diff_nn]))

        ax1.scatter(pfratio, diff_nonlinear, s=3, c='r', marker="o", label='nonlinear', alpha=0.5)
        ax1.scatter(pfratio, diff_loglinear, s=3, c='b', marker="s", label='loglinear', alpha=0.5)
        ax1.scatter(pfratio, diff_nn, s=3, c='k', marker="*", label='neural network', alpha=0.5)

        plt.title(model_name + ' (' + str(n_features) + ' Features)')
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

        print('Upper value:', (md + 1.96 * sd))
        print('Lower value:', (md - 1.96 * sd))
        print('Mean value:', md)

        plt.xlabel('Measured PF')
        plt.ylabel('Measured PF - Imputed PF')
        plt.legend(loc='upper left')
        plt.savefig('Figures/measured_pf_and_imputed_pf_' + str(n_features) + '_FilterSpO2_' + str(filterSpO2) + '.png')
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',dest='data', type=str,
                        default='new_data.csv', help='Path to data file')
    parser.add_argument('--model', dest='model', type=str,
                        default='sgd', help='model to run: linear_regression, sgd, neural_network, svr')
    parser.add_argument('--seven_features', dest='seven_features', type=bool,
                        default=False, help='If true, use 7 input features, if false, use 3 input features')
    parser.add_argument('--fitlerSpO2', dest='filterSpO2', type=bool,
                        default=True, help='Whether or not to filter samples with SpO2 value greater than 96')
    parser.add_argument('--load_model', dest='load_model', type=bool,
                        default=False, help='Whether or not to load pretrained model')  # model path is hard coded
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = args.data
    model = args.model
    seven_features = args.seven_features
    fitlerSpO2 = args.filterSpO2
    load_model = args.load_model

    #run_predictor(path, model=model, seven_features=seven_features, fitlerSpO2=fitlerSpO2, load_model=load_model)

    run_predictor(path, model='neural_network', seven_features=False, filterSpO2=True, load_model=False)


if __name__ == '__main__':
    main()
