import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import scipy

feature_columns = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c', 'precipitation_amt_mm', 'ndvi_ne', 'ndvi_nw', 'ndvi_se','ndvi_sw', 'week_start_date']
features = pd.read_csv('dengue_features_train.csv', usecols=feature_columns)

#removing categorical variables
features = pd.get_dummies(features)
#fill the missing values with the mean of the column
features.fillna(features.mean(), inplace=True)

labels = pd.read_csv('dengue_labels_train.csv')
labels = labels.drop('city', axis = 1)
labels = labels.drop('year', axis = 1)
labels = labels.drop('weekofyear', axis = 1)

test_features = pd.read_csv('dengue_features_test.csv')
test_features= test_features.drop('week_start_date', axis = 1)
test_features= test_features.drop('year', axis = 1)

test_features= test_features.drop('reanalysis_air_temp_k', axis = 1)
test_features= test_features.drop('reanalysis_max_air_temp_k', axis = 1)
test_features= test_features.drop('reanalysis_min_air_temp_k', axis = 1)
test_features= test_features.drop('reanalysis_precip_amt_kg_per_m2', axis = 1)
test_features= test_features.drop('precipitation_amt_mm', axis = 1)
test_features= test_features.drop('reanalysis_relative_humidity_percent', axis = 1)
test_features= test_features.drop('station_max_temp_c', axis = 1)
test_features= test_features.drop('station_min_temp_c', axis = 1)
test_features= test_features.drop('station_diur_temp_rng_c', axis = 1)
test_features= test_features.drop('station_precip_mm', axis = 1)
test_features= test_features.drop('reanalysis_dew_point_temp_k', axis = 1)

#removing categorical variables
test_features = pd.get_dummies(test_features)
#fill the missing values with the mean of the column
test_features.fillna(test_features.mean(), inplace=True)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(features, labels)

predictions = rf.predict(test_features)

results = [round(x) for x in list(predictions)]
for i in results:
    print(i)
