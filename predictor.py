
import tkinter as tk
import numpy as np
from joblib import load
import keras
from keras import backend
import sys, os
import warnings
import multiprocessing

warnings.simplefilter(action='ignore', category=FutureWarning)
multiprocessing.freeze_support()

try:
   wd = sys._MEIPASS
except AttributeError:
   wd = os.getcwd()


'''
    Define GUI
'''

root= tk.Tk()
root.title("PaO2 Predictor")

# Add a grid
mainframe = tk.Frame(root)
mainframe.grid(row=18, column=3, sticky=(tk.N, tk.W, tk.E, tk.S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 100, padx = 100)


# dropdown for selecting model
type_choices = ['Linear_Regression',
                'svr',
                'Neural_Network']

tkvar = tk.StringVar(root)
tkvar.set('Linear_Regression') # set the default option

popupMenu = tk.OptionMenu(mainframe, tkvar,  *type_choices)
tk.Label(mainframe, text="Choose a model").grid(row=1, column =0)
popupMenu.grid(row=1, column=1)

# checkbox for deciding whether or not filter
filter_var = tk.IntVar()
cb_filter = tk.Checkbutton(mainframe, text='Model trained with SpO2 >= 96 samples excluded', variable=filter_var, onvalue=1, offvalue=0)
cb_filter.grid(row=3, column=1)

tk.Label(mainframe, text="Input Features").grid(row=4, column=0)

# Input Features:
curr_row = 5
tk.Label(mainframe, text="SpO2 (0 ~ 100):").grid(row=curr_row, column=0)
entry_spo2 = tk.Entry(mainframe)
entry_spo2.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="FiO2 (0.0 ~ 1.0):").grid(row=curr_row, column=0)
entry_fio2 = tk.Entry(mainframe)
entry_fio2.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="PEEP (cmH₂O):").grid(row=curr_row, column=0)
entry_peep = tk.Entry(mainframe)
entry_peep.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="Tidal Volumne (ml):").grid(row=curr_row, column=0)
entry_vt = tk.Entry(mainframe)
entry_vt.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="MAP:").grid(row=curr_row, column=0)
entry_map = tk.Entry(mainframe)
entry_map.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="Temperature (celsius):").grid(row=curr_row, column=0)
entry_temp = tk.Entry(mainframe)
entry_temp.grid(row=curr_row, column=1)
curr_row += 1

tk.Label(mainframe, text="vaso (1 or 0):").grid(row=curr_row, column=0)
entry_vaso = tk.Entry(mainframe)
entry_vaso.grid(row=curr_row, column=1)
curr_row += 1

# Feature used:
feature_3_var = tk.IntVar()
cb_features = tk.Checkbutton(mainframe, text='Use Top 3 Features (SpO2, FiO2, Peep)', variable=feature_3_var, onvalue=1, offvalue=0)
cb_features.grid(row=curr_row, column=1)
curr_row += 1

# Initialize Predict label
pred_var = tk.StringVar('')
pred_label = tk.Label(mainframe, textvariable=pred_var, fg='green', font=('helvetica', 20, 'bold'))

'''
    Prediction functions
'''
# define prediction behavior
def predict():
    tk.Label(mainframe, text='                                ').grid(row=14, column=1) # Clear text
    in_features = read_input()
    model = load_model()
    input_scaler, output_scaler = load_scalers()

    # 3 features scaling
    if feature_3_var.get() == 1:
        in_features = np.expand_dims(in_features, axis=0)
        in_features = input_scaler.transform(in_features)
    else:
        # 7 feature scaling
        other_features = in_features[0:-1] # exclude vaso feature
        other_features = np.expand_dims(other_features, axis=0)
        other_features = input_scaler.transform(other_features).squeeze()
        in_features[0:-1] = other_features
        in_features = np.expand_dims(in_features, axis=0)

    pred = model.predict(in_features)

     # Regression Model predicts PaO2 Value
    pred = np.expand_dims(pred, axis=0)
    pred = output_scaler.inverse_transform(pred).squeeze()
    pred = np.round(pred, decimals=2)
    pred_var.set('Predicted PaO2 Value: ' + str(pred))
    print('Prediction PaO2 Value:', pred)


    root.update()
    pred_label.grid(row=14, column=0)
    return pred

# Load model
def load_model():
    model_dir = 'saved_models/'
    model_name = tkvar.get()

    model_dir += 'regressor/'
    model_dir += 'FilteredSpO2/' if filter_var.get() == 1 else 'NoSpO2Filtering/'

    model_surfix = '_3_features' if feature_3_var.get() == 1 else '_7_features'
    model_name = model_name.split(' ')[0]
    model_path = model_dir + model_name + model_surfix

    if 'Neural_Network' not in tkvar.get():
        model = load(os.path.join(wd, model_path + '.joblib'))
    else:
        model = keras.models.load_model(os.path.join(wd, model_path + '.ckpt'), custom_objects={'rmse': rmse})
    return model


# Read input entries
def read_input():
    inputs = []
    inputs.append(float(entry_spo2.get()))
    inputs.append(float(entry_fio2.get()))
    inputs.append(float(entry_peep.get()))
    if feature_3_var.get() == 0:
        inputs.append(float(entry_vt.get()))
        inputs.append(float(entry_map.get()))
        inputs.append(float(entry_temp.get()))
        inputs.append(float(entry_vaso.get()))
    return np.array(inputs)


# Load pretrained scalers
def load_scalers():
    scaler_dir = 'saved_models/'
    scaler_dir += 'regressor/'
    scaler_dir += 'FilteredSpO2/' if filter_var.get() == 1 else 'NoSpO2Filtering/'

    in_scaler_name = 'InputScaler_'
    in_scaler_name += '3_features' if feature_3_var.get() == 1 else '7_features'

    out_scaler_name = 'OutputScaler_'
    out_scaler_name += '3_features' if feature_3_var.get() == 1 else '7_features'

    print(in_scaler_name, out_scaler_name)

    input_scaler = load(os.path.join(wd, scaler_dir + in_scaler_name + '.joblib'))
    output_scaler = load(os.path.join(wd, scaler_dir + out_scaler_name + '.joblib'))
    return input_scaler, output_scaler

def rmse(y_true,y_predict):
    return backend.sqrt(backend.mean(backend.square(y_predict-y_true),axis=-1))

'''
    Button Action
'''
# Run botton:
btn_go = tk.Button(mainframe, text='Predict', command=predict, bg='green',fg='black')
btn_go.grid(row=curr_row, column= 2)
root.mainloop()


