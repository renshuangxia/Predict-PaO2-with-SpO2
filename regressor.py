import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection,  linear_model, svm
from tensorflow import set_random_seed
import argparse
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)
set_random_seed(1)

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


def run_predictor(path, model='linear_regression', seven_features=False, fitlerSpO2=False):
    data = init_data(path, seven_features=seven_features, fitlerSpO2=fitlerSpO2)
    X = data[:, 0:-1]
    y = data[:, -1]

    stratifier = model_selection.StratifiedKFold(n_splits=10, shuffle=False)

    rmse_results = []
    total_inversed_preds = []
    i = 1
    for train_index, test_index in stratifier.split(X, y):
        print("Current fold:",i)
        i += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_scale, X_test_scale, y_train_scale, y_test_scale, output_scaler = normalize(X_train, X_test, y_train, y_test, seven_features=seven_features)

        y_train_scale = np.array(y_train_scale).squeeze()
        y_test_scale = np.array(y_test_scale).squeeze()

        if model == 'svr':
            predictor = svm.SVR(kernel='rbf')
        elif model == 'neural_network':
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
        elif model == 'sgd':
            predictor = linear_model.SGDRegressor()
        else:
            predictor = linear_model.LinearRegression()
        print('start training...')
        if model == 'neural_network':
            predictor.fit(X_train_scale, y_train_scale, epochs=50, batch_size=50)
        else:
            predictor.fit(X_train_scale, y_train_scale)
        test_pred = predictor.predict(X_test_scale)

        inverse_test_pred = output_scaler.inverse_transform(test_pred)
        total_inversed_preds.append(inverse_test_pred)

        rmse_result = np_rmse(np.array(inverse_test_pred).flatten(), np.array(y_test).flatten())
        rmse_results.append(rmse_result)

        print(model + ' rmse:', rmse_result)

    loglinear_rmse, nonlinear_rmse, pfratio, pred_pfratio_loglinear, pred_pfratio_nonlinear = get_baseline(X, y)
    print('\n========================')
    print('         RESULTS')
    print('------------------------')
    print('total cases:', X.shape[0])
    print("loglinear_rmse,", loglinear_rmse)
    print("nonlinear_rmse", nonlinear_rmse)
    mean_rmse = np.mean(np.array(rmse_results))
    print(model + ' mean rmse:', mean_rmse)
    print('========================')
    total_inversed_preds = np.array(total_inversed_preds)

    # plot
    print("plot results...")
    pred_pao2_nn = total_inversed_preds.flatten()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    test_fio2 = X[:, 1]
    pred_pfratio_nn = pred_pao2_nn / test_fio2

    ax1.scatter(pfratio, pfratio - pred_pfratio_nonlinear, s=2, c='r', marker="o", label='nonlinear', alpha=0.5)
    ax1.scatter(pfratio, pfratio - pred_pfratio_loglinear, s=2, c='b', marker="s", label='loglinear', alpha=0.5)
    ax1.scatter(pfratio, pfratio - pred_pfratio_nn, s=2, c='y', marker="*", label='neural network', alpha=0.2)

    plt.xlabel('Measured PF')
    plt.ylabel('Measured PF - Imputed PF')
    plt.legend(loc='upper left')
    plt.savefig('measured_pf_and_imputed_pf.png')
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = args.data
    model = args.model
    seven_features = args.seven_features
    fitlerSpO2 = args.filterSpO2
    run_predictor(path, model=model, seven_features=seven_features, fitlerSpO2=fitlerSpO2)


if __name__ == '__main__':
    main()
