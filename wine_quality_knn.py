import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split

# Load dataset
winedata = np.loadtxt('winedata.csv', delimiter=",", skiprows=1)

# Set print options for readability
np.set_printoptions(precision=3, suppress=True)

# Define feature matrix (X) and target variable (y)
X = winedata[:, 0:11]  # First 11 columns as features
y = winedata[:, 11]  # Last column as target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale feature values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler to training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Define KNN classifier and train
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train.ravel())  # Ensure y_train is 1D

# Define vineyard samples
hpv = np.array([7.9, 0.18, 0.04, 19.5, 0.044, 47, 97, 0.9938, 3.14, 0.42, 10.1]).reshape(1, -1)
htc = np.array([6.6, 0.22, 0.15, 4.2, 0.35, 35, 138, 0.8750, 3.05, 0.63, 8.1]).reshape(1, -1)
olv = np.array([8.1, 0.35, 0.14, 1.5, 0.37, 45, 132, 0.8850, 3.75, 0.55, 9.5]).reshape(1, -1)

# Scale vineyard samples using the SAME scaler
scaled_hpv = scaler.transform(hpv)
scaled_htc = scaler.transform(htc)
scaled_olv = scaler.transform(olv)

# Make predictions
print("Predicted quality for OLV Vineyard:", clf.predict(scaled_olv)[0])
