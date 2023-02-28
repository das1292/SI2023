import pandas as pd
import numpy as np
from scipy.stats import mode

# Zadanie4
# a
# Tworzenie przykładowych danych
data = np.random.rand(100, 5) * 10
data[:10, :] = np.nan
print(data[:15, :])

# Generowanie 10% wartości nieznanych
missing_ratio = 0.1
missing_mask = np.random.choice([True, False], size=data.shape, p=[missing_ratio, 1 - missing_ratio])

# Naprawianie wartości nieznanych metodą szukania najczęściej występującej wartości
for i in range(data.shape[1]):
    col = data[:, i]
    mode_val = mode(col, nan_policy='omit', axis=0, keepdims=True)[0][0]
    col[np.isnan(col)] = mode_val
    data[:, i] = col

print(data[:15, :])

# b
# Tworzenie przykładowych danych
data = np.random.rand(100, 5) * 10

# Normalizacja na przedział <a,b>
a = -1
b = 1
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

# Normalizacja na przedział <0,1>
a = 0
b = 1
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

# Normalizacja na przedział <-10,10>
a = -10
b = 10
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

# d
# Load the dataset
data = pd.read_csv('dane/Churn_Modelling.csv')

# Drop the unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Age'], axis=1)

# Convert categorical variable "Geography" to dummy variables
geography_dummies = pd.get_dummies(data['Geography'], drop_first=True)
data = pd.concat([data, geography_dummies], axis=1)
data = data.drop(['Geography'], axis=1)

# Display the converted data
print(data.head())