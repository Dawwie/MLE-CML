import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as rmse
from matplotlib import pyplot as plt
reg = LinearRegression()

X_train = np.genfromtxt("data/train_features.csv").reshape(-1, 1)
y_train = np.genfromtxt("data/train_labels.csv").reshape(-1, 1)
X_test = np.genfromtxt("data/test_features.csv").reshape(-1, 1)
y_test = np.genfromtxt("data/test_labels.csv").reshape(-1, 1)

lin_reg = reg.fit(X_train, y_train)
score = lin_reg.score(X_test, y_test)
prediction = lin_reg.predict(X_test)
rmse = rmse(y_test, prediction, squared=False)

print("Score:", score)
print("RMSE:", rmse)

with open("results.txt", 'w') as outfile:
    outfile.write("Score: " + str(score) + "\n")
    outfile.write("RMSE: " + str(rmse) + "\n")

plt.figure(figsize=(10, 6))
m, b = np.polyfit(X_train.ravel(), y_train.ravel(), 1)
plt.plot(X_train, m*X_train + b)
plt.plot(X_train, y_train, 'o')
plt.savefig('lin_reg_plot.png')