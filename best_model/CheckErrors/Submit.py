import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

def preprocess_data(data_path, labels_path=None):
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c', 'precipitation_amt_mm', 'ndvi_ne', 'ndvi_nw', 'ndvi_se','ndvi_sw', 'week_start_date']
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    df['station_avg_temp_c'] = df['station_avg_temp_c'].rolling(window=5).mean()
    df['precipitation_amt_mm'] = df['precipitation_amt_mm'].rolling(window=5).mean()

    df.fillna(method='ffill', inplace=True)
    df = df.fillna(df.mean())

    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    for i in range(1, 5):
        df['quarter_' + str(i)] = df['week_start_date'].apply(lambda date: 1 if (
            ((i - 1) * 3 < date.month) and (date.month <= i * 3)) else 0)
        features.append('quarter_' + str(i))

    df = df.drop(['week_start_date'], axis=1)
    features.remove('week_start_date')
    df = df[features]
    sj_label = None
    iq_label = None
    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2]).loc[df.index]
        sj_label = pd.DataFrame(labels.loc['sj'])
        iq_label = pd.DataFrame(labels.loc['iq'])

    sj = pd.DataFrame(df.loc['sj'])
    iq = pd.DataFrame(df.loc['iq'])

    return sj, iq, sj_label, iq_label

sj_train, iq_train, sj_label, iq_label = preprocess_data('dengue_features_train.csv', 'dengue_labels_train.csv')

sj_train_features, sj_test_features, sj_train_labels, sj_test_labels = train_test_split(
    sj_train, sj_label['total_cases'], test_size=0.25, random_state=0, shuffle=False)

iq_train_features, iq_test_features, iq_train_labels, iq_test_labels = train_test_split(
    iq_train, iq_label['total_cases'], test_size=0.25, random_state=0, shuffle=False)

sj_model = RandomForestRegressor(n_estimators=1000, max_features='auto',
                                 max_depth=10, min_samples_leaf=0.005,
                                 criterion='mae', min_weight_fraction_leaf=0.1
                                , warm_start=True)

sj_model.fit(sj_train_features, sj_train_labels)
sj_pred_val = sj_model.predict(sj_test_features)

print("SJ   " + str(mean_absolute_error(sj_test_labels, sj_pred_val)))

iq_model = RandomForestRegressor(n_estimators=1000, max_features='auto',
                                 max_depth=10, min_samples_leaf=0.005,
                                 criterion='mae', min_weight_fraction_leaf=0.1
                                , warm_start=True)

iq_model.fit(iq_train_features, iq_train_labels)
iq_pred_val = iq_model.predict(iq_test_features)

print("IQ   " + str(mean_absolute_error(iq_test_labels, iq_pred_val)))

figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
pd.DataFrame(sj_pred_val).plot(ax=axes[0], label="Predictions")
sj_test_labels.plot(ax=axes[0], label="Actual")

# plot iq
pd.DataFrame(iq_pred_val).plot(ax=axes[1], label="Predictions")
iq_test_labels.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

# labels = np.array(labels['total_cases'])
#
# feature_list = list(features.columns)
#
# features = np.array(features)
#
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# # Train the model on training data
# rf.fit(train_features, train_labels)
#
# predictions = rf.predict(test_features)
#
# results = [round(x) for x in list(predictions)]
#
# errors = abs(results - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'cases.')

sj_test, iq_test, sj_test_label, iq_test_label = preprocess_data('dengue_features_test.csv')

sj_predictions = sj_model.predict(sj_test).astype(int)
iq_predictions = iq_model.predict(iq_test).astype(int)

submission = pd.read_csv("submission_format.csv", index_col=[0, 1, 2])

# sj_predictions = np.array([adder(xi) for xi in sj_predictions])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])

submission.to_csv("2_submission_latest.csv")