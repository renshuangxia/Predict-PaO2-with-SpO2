# Predict-PaO2-with-SpO2
This project aims to predict PaO2 values based on SpO2 and other patient data. The dataset used are extracted from MIMIC III database. There are two scripts in this project:

1.  Regressor: Predict PaO2 values based on input features.
2.  Classifier:	Predict PaO2/SpO2 ratio that falls into two categories:
  
    - Class I: PaO2/SpO2 ratio <= 150
  
    - Class II: PaO2/SpO2 ratio > 150


# Input features:

Both scripts use two different sets of input features:
•	SpO2, FiO2, Peep, Vt, Map, Temperature, vaso
•	SpO2, FiO2, Peep

Above features were extracted from deidentified patient records in MIMIC III database.

# Regressor:

The regressor can predict PaO2 values with the following models:
1.	sklearn LinearRegression
2.	sklearn SVR
3.	Neural Network (tensorflow)


# Classifier:

The classifier can predict PaO2/SpO2 ratio class with the following models:
1.	sklearn LogisticRegression
2.	sklearn SVC
3.	sklearn MLPClassifier


# Run scripts:

Both scripts contain the same set of four arguments:

--data: relative path to the data file (in csv format)

--model: select model to use. For regressor, select from (‘linear_regression’, ‘sgd’, ‘neural_network’, ‘svr’). For classifier, select from (‘logistic_regression’, ‘sgd’, ‘mlp’, ‘svc’)

--seven_features: if set to true, use the seven input features as described above. Otherwise use three features.

--fitlerSpO2: if set to true, exclude samples with SpO2 values greater than 96.
	
# Apps:

The apps can be downloaded to add inputs and see outputs of different models.
Download address:

https://drive.google.com/drive/folders/1AoieWO0w3BXvEpw6c0-OomjeQHzFJXY_?usp=sharing

Windows: PaO2Predictor.exe

Mac ios: PaO2Predictor_Mac.zip (need to unzip this)
