import numpy as np
import pandas as pd

arr = np.array([5, 2, 9, 1, 5])
sorted_indexes = np.argsort(arr)
print(sorted_indexes)

field_data = pd.read_csv('./data/field_data.csv', sep=';')
RS_data = pd.read_csv('./data/RS_features.csv', sep=';')

combined_data = pd.merge(field_data, RS_data, on='id', how='left')
combined_data.to_csv('a.csv', sep=';')

data_features = ['h0f', 'h5f', 'h10f', 'h20f', 'h30f', 'h40f','h50f', 'h60f', 'h70f', 'h80f', 'h85f', 'h90f',	'h95f',	'h100f', 'vegf', 'h_mean', 'vkph_ka']
data_set = combined_data[data_features].dropna()
data_set.to_csv('data_set.csv', sep=';')

data_set_numpy = data_set.values
input_data = data_set_numpy[:,:-1]
output_data = data_set_numpy[:,-1]
print(input_data.shape, output_data.shape)