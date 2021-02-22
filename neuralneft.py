# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:31:45 2020

@author: SHER_CODE
"""


#%%

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%%

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

#%%

features = pd.read_csv(r'oil\фьючерсы.csv', sep = ';')[::-1] # https://ru.investing.com/commodities/brent-oil-historical-data
features['Дата'] = pd.to_datetime(features['Дата'])

#%%

features[::-1].head()

#%%

features_2 = pd.read_csv(r'oil\features_2.csv', sep = ',')
del features_2['Unnamed: 0']
features_2['Date'] = pd.to_datetime(features_2['Date'])

#%%

features_2.head()

#%%

print(len(features), len(features_2))

#%%

for i in range(len(features['Объём'])):
    try:
        features['Объём'][i] = float(features['Объём'][i])
    except:
        features['Объём'][i] = features['Объём'][i-1]

#%%

features['Цена'] = pd.to_numeric(features['Цена'])
features['Откр.'] = pd.to_numeric(features['Откр.'])
features['Макс.'] = pd.to_numeric(features['Макс.'])
features['Мин.'] = pd.to_numeric(features['Мин.'])
features['Объём'] = pd.to_numeric(features['Объём'])
features['Изм.'] = pd.to_numeric(features['Изм.'])

features_2['Oil'] = pd.to_numeric(features_2['Oil'])
features_2['Gas'] = pd.to_numeric(features_2['Gas'])
features_2['Misc'] = pd.to_numeric(features_2['Misc'])
features_2['Total'] = pd.to_numeric(features_2['Total'])

#%%

features.rename(columns={'Дата': 'Date'}, inplace=True)

#%%

features.head()

#%%

features_2.head()

#%%

result = features.merge(features_2,
                        how='left',
                        on='Date') 

#%%

result

last_date = result['Date'][len(result)-1]
#%%

features = result[['Цена', 'Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм.', 'Oil', 'Gas', 'Misc', 'Total', 'Oil.1', 'Gas.1']]

#%%

features

#%%

def fill_na(name):
    number = 0
    for i in range(len(features)):
        if not math.isnan(features[name][i]):
            number = features[name][i]
        if math.isnan(features[name][i]):
            features[name][i] = number
            
#%%

fill_na('Oil')
fill_na('Gas')
fill_na('Misc')
fill_na('Total')
fill_na('Oil.1')
fill_na('Gas.1')

#%%

features

#%%

TRAIN_SPLIT = 3000

#%%

features[['Цена', 'Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм.']].plot(subplots=True)

#%%

features[['Oil', 'Gas', 'Misc', 'Total', 'Oil.1', 'Gas.1']].plot(subplots=True)

#%%

scaler = StandardScaler()

dataset = features.values

scaler = scaler.fit(dataset)
dataset = scaler.transform(dataset)

scaler.inverse_transform
# inversed = scaler.inverse_transform(standardized)

#%%

past_history = 700
future_target = 372

STEP = 6
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

#%%

print ('Один фрагмент данных: {}\n'.format(x_train_multi[0].shape))
print ('Целевые данные для предсказания: {}'.format(y_train_multi[0].shape))

#%%

BATCH_SIZE = 512
BUFFER_SIZE = 10000

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%%

def create_time_steps(length):
    return list(range(-length, 0))

#%%

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

#%%

for x, y in train_data_multi.take(1):
    
    
    history = np.array(x[0])*scaler.scale_[0] + scaler.mean_[0]
    true_future = np.array(y[0])*scaler.scale_[0] + scaler.mean_[0]
    prediction = np.array([0])
    
    multi_step_plot(history, true_future, prediction)

#%%

multi_step_model = tf.keras.models.Sequential()

multi_step_model.add(tf.keras.layers.LSTM(256,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:],
                                          activation='tanh',
                                          recurrent_activation="sigmoid",
                                          recurrent_dropout=0,
                                          unroll=False,
                                          use_bias=True))

multi_step_model.add(tf.keras.layers.Dropout(0.2))
                  
multi_step_model.add(tf.keras.layers.LSTM(512,
                                          return_sequences=False,
                                          activation='tanh',
                                          recurrent_activation="sigmoid",
                                          recurrent_dropout=0,
                                          unroll=False,
                                          use_bias=True))

                     
multi_step_model.add(tf.keras.layers.Dropout(0.2))

multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

#%%

for x, y in val_data_multi.take(1):
    print (multi_step_model.predict(x).shape)

#%%

EPOCHS = 50

EVALUATION_INTERVAL = 200

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)


#%%

import tensorflow as tf

#multi_step_model = tf.keras.models.load_model(r'D:\saved_models\brent.h5')

#%%

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Потери на этапе обучения')
    plt.plot(epochs, val_loss, 'r', label='Потери на этапе проверки')
    plt.title(title)
    plt.legend()

    plt.show()

#%%

plot_train_history(multi_step_history, 'Интервальное прогнозирование: потери на этапах обучения и проверки')

#%%

for x, y in val_data_multi.take(3):
    
    history = np.array(x[0])*scaler.scale_[0] + scaler.mean_[0]
    true_future = np.array(y[0])*scaler.scale_[0] + scaler.mean_[0]
    prediction = np.array(multi_step_model.predict(x)[0])*scaler.scale_[0] + scaler.mean_[0]
    
    multi_step_plot(history, true_future, prediction)

#%%
start = last_date

d = {'Дата':list(pd.date_range(start, periods=372, freq="D"))}
d1 = {'Цена': prediction}

part_1 = pd.Series(data=d)
part_2 = pd.Series(data=d1)

result = pd.DataFrame({'Дата': part_1['Дата'], 'Цена': part_2['Цена']})

result.to_csv(r'oil\prediction.csv', encoding = 'utf-8')

#%%

multi_step_model.save(r'model\brent.h5')

#%%


#%%



























