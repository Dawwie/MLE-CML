import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read dataset
df = pd.read_csv(
    'country_vaccinations.csv', header=0,
    usecols=['total_vaccinations', 'daily_vaccinations', 'people_fully_vaccinated']).dropna()

# prepare features and labales
X = df['total_vaccinations'].to_numpy()
y = df['daily_vaccinations'].to_numpy()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=777)

# create folder with data if not exist
if not os.path.isdir("data"):
    os.mkdir("data")

# save splited data
np.savetxt("data/train_features.csv", X_train, delimiter=",")
np.savetxt("data/test_features.csv", X_test, delimiter=",")
np.savetxt("data/train_labels.csv", y_train, delimiter=",")
np.savetxt("data/test_labels.csv", y_test, delimiter=",")
