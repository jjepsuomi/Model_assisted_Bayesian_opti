import numpy as np
import pandas as pd

arr = np.array([5, 2, 9, 1, 5])
sorted_indexes = np.argsort(arr)
print(sorted_indexes)

field_data = pd.read_csv('./data/field_data.csv', sep=';')
total_volume_data = field_data["vkph_ka"]

RS_features = ['h0f', 'h5f', 'h10f', 'h20f', 'h30f', 'h40f','h50f', 'h60f', 'h70f', 'h80f', 'h85f', 'h90f',	'h95f',	'h100f', 'vegf', 'h_mean']

RS_data = pd.read_csv('./data/RS_features.csv', sep=';')
feature_data = RS_data[RS_features]
print(feature_data)