from numpy import array
from scipy import io
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import os
import time
import h5py
import joblib
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = 0

if dataset == 0:
    data_input = h5py.File('ML_datasets\data_zern_coeffs_timestep_train.mat','r').get('zern_coeff_timestep_concat_train')
    data_output = h5py.File('ML_datasets\data_voltages_timestep_train.mat','r').get('voltages_timestep_concat_train')
    data_input = np.array(data_input).T
    data_output = np.array(data_output).T
    test_input = h5py.File('ML_datasets\data_zern_coeffs_timestep_test.mat','r').get('zern_coeff_timestep_concat_test')
    test_output = h5py.File('ML_datasets\data_voltages_timestep_test.mat','r').get('voltages_timestep_concat_test')
    test_input = np.array(test_input).T
    test_output = np.array(test_output).T
    sample_num = 32000
    test_num = 4000

# Define parameters
timestep_num = 3
feature_num = 6
actuator_num = 69
repeat_num = 1
param_num = 1
unit_num_1 = 128
epochs_num = 30
batch_size_num = [16]
layer_num = [4]

# Create training input dataset
# print(np.argwhere(np.isnan(data_input[:,0])))
X1 = data_input[:,0].reshape(-1,1)
scaler_input1 = MinMaxScaler(feature_range = (-1, 1))
scaler_input1 = scaler_input1.fit(X1)
print('Min1: %.3f, Max1: %.3f' % (scaler_input1.data_min_, scaler_input1.data_max_))
X1 = scaler_input1.transform(X1)
joblib.dump(scaler_input1, 'model_param\scaler_input1.gz')

# print(np.argwhere(np.isnan(data_input[:,1])))
X2 = data_input[:,1].reshape(-1,1)
scaler_input2 = MinMaxScaler(feature_range = (-1, 1))
scaler_input2 = scaler_input2.fit(X2)
print('Min2: %.3f, Max2: %.3f' % (scaler_input2.data_min_, scaler_input2.data_max_))
X2 = scaler_input2.transform(X2)
joblib.dump(scaler_input2, 'model_param\scaler_input2.gz')

# print(np.argwhere(np.isnan(data_input[:,2])))
X3 = data_input[:,2].reshape(-1,1)
scaler_input3 = MinMaxScaler(feature_range = (-1, 1))
scaler_input3 = scaler_input3.fit(X3)
print('Min3: %.3f, Max3: %.3f' % (scaler_input3.data_min_, scaler_input3.data_max_))
X3 = scaler_input3.transform(X3)
joblib.dump(scaler_input3, 'model_param\scaler_input3.gz')

# print(np.argwhere(np.isnan(data_input[:,3])))
X4 = data_input[:,3].reshape(-1,1)
scaler_input4 = MinMaxScaler(feature_range = (-1, 1))
scaler_input4 = scaler_input4.fit(X4)
print('Min4: %.3f, Max4: %.3f' % (scaler_input4.data_min_, scaler_input4.data_max_))
X4 = scaler_input4.transform(X4)
joblib.dump(scaler_input4, 'model_param\scaler_input4.gz')

# print(np.argwhere(np.isnan(data_input[:,3])))
X5 = data_input[:,4].reshape(-1,1)
scaler_input5 = MinMaxScaler(feature_range = (-1, 1))
scaler_input5 = scaler_input5.fit(X5)
print('Min5: %.3f, Max5: %.3f' % (scaler_input5.data_min_, scaler_input5.data_max_))
X5 = scaler_input5.transform(X5)
joblib.dump(scaler_input5, 'model_param\scaler_input5.gz')

# print(np.argwhere(np.isnan(data_input[:,3])))
X6 = data_input[:,5].reshape(-1,1)
scaler_input6 = MinMaxScaler(feature_range = (-1, 1))
scaler_input6 = scaler_input6.fit(X6)
print('Min6: %.3f, Max6: %.3f' % (scaler_input6.data_min_, scaler_input6.data_max_))
X6 = scaler_input6.transform(X6)
joblib.dump(scaler_input6, 'model_param\scaler_input6.gz')

X = np.column_stack((X1, X2, X3, X4, X5, X6))
X_train = array(X).reshape(sample_num, timestep_num, feature_num)

# Create test input dataset
# print(np.argwhere(np.isnan(test_input[:,0])))
X1_test = test_input[:,0].reshape(-1,1)
X1_test = scaler_input1.transform(X1_test)

# print(np.argwhere(np.isnan(test_input[:,1])))
X2_test = test_input[:,1].reshape(-1,1)
X2_test = scaler_input2.transform(X2_test)

# print(np.argwhere(np.isnan(test_input[:,2])))
X3_test = test_input[:,2].reshape(-1,1)
X3_test = scaler_input3.transform(X3_test)

# print(np.argwhere(np.isnan(test_input[:,3])))
X4_test = test_input[:,3].reshape(-1,1)
X4_test = scaler_input4.transform(X4_test)

# print(np.argwhere(np.isnan(test_input[:,4])))
X5_test = test_input[:,4].reshape(-1,1)
X5_test = scaler_input5.transform(X5_test)

# print(np.argwhere(np.isnan(test_input[:,5])))
X6_test = test_input[:,5].reshape(-1,1)
X6_test = scaler_input6.transform(X6_test)

X_test = np.column_stack((X1_test, X2_test, X3_test, X4_test, X5_test, X6_test))
X_test = array(X_test).reshape(test_num, timestep_num, feature_num)

# X_train = X_train[0:5000, :]
# X_test = X_test[0:2000, :]
# X_train = X[0:sample_num - test_num, :]
# X_test = X[sample_num - test_num:, :]

# Create training output dataset
Y = data_output.reshape(-1,1)
scaler_output = MinMaxScaler(feature_range = (-1, 1))
scaler_output = scaler_output.fit(Y)
print('Min: %.3f, Max: %.3f' % (scaler_output.data_min_, scaler_output.data_max_))
Y = scaler_output.transform(Y)
joblib.dump(scaler_output, 'model_param\scaler_output.gz')
Y_train = array(Y).reshape(sample_num, actuator_num)

# Create test output dataset
Y_test = test_output.reshape(-1,1)
Y_test = scaler_output.transform(Y_test)
Y_test = array(Y_test).reshape(test_num, actuator_num)

# Y_train = Y_train[0:5000, :]
# Y_test = Y_test[0:2000, :]
# Y_train = Y[0:sample_num - test_num, :]
# Y_test = Y[sample_num - test_num:, :]

# Collect data across multiple repeats
train = DataFrame()
val = DataFrame()
rmse = np.zeros([repeat_num, param_num])
rmse_raw = np.zeros([repeat_num, test_num])

for j in range(param_num):

    for i in range(repeat_num):

        # Define stacked LSTM model
        model = Sequential()
        model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform', return_sequences = True), input_shape = (timestep_num, feature_num)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform', return_sequences = True)))
        # model.add(BatchNormalization())
        # model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform', return_sequences = True)))
        # model.add(BatchNormalization())
        # model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform', return_sequences = True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform', return_sequences = True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(unit_num_1, activation = 'relu', kernel_initializer = 'he_uniform')))
        model.add(Dense(actuator_num, activation = 'linear', kernel_initializer = 'he_uniform'))

        # Define learning rate schedule
        # lr_schedule = ExponentialDecay(initial_learning_rate = 1e-3, decay_steps = 10000, decay_rate = 0.9)
        # optimizer_with_decay = SGD(learning_rate = lr_schedule)
        lr_schedule = ExponentialDecay(initial_learning_rate = 1e-3, decay_steps = 10000, decay_rate = 0.9)
        optimizer_with_decay = Adam(learning_rate = lr_schedule)
        # optimizer_with_decay = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7, decay = 0.1)
        # optimizer_with_decay = RMSprop(learning_rate = 1e-3, rho = 0.9, momentum = 0.8, epsilon = 1e-07)

        # Define loss and optimizer
        model.compile(optimizer = optimizer_with_decay, loss = 'mse', metrics = [RootMeanSquaredError()])
        print(model.summary())

        # Fit the model
        t1 = time.perf_counter()
        history = model.fit(X_train, Y_train, epochs = epochs_num, validation_split = 0.2, verbose = 1, batch_size = batch_size_num[j], shuffle = True)
        t2 = time.perf_counter()
        print('Training time = {}'.format(t2-t1))

        # Story history
        train[str(i)] = history.history['loss']
        val[str(i)] = history.history['val_loss']

        # Make a prediction
        test_output = model.predict(X_test, verbose = 1)
        test_output_inversed = scaler_output.inverse_transform(test_output)
        # print('Test output inversed: {}'.format(np.round(test_output_inversed, 3)))
        # print('Test output inversed shape: {}'.format(np.shape(test_output_inversed)))

        # Calculate RMSE
        Y_test_inversed = scaler_output.inverse_transform(Y_test)
        # print('Test target inversed: {}'.format(np.round(Y_test_inversed, 3)))
        # print('Test target inversed shape: {}'.format(np.shape(Y_test_inversed)))
        rmse[i, j] = mean_squared_error(Y_test_inversed.T, test_output_inversed.T, squared = False)
        print('Test RMSE: {}'.format(np.round(rmse[i, j], 3)))
        rmse_raw[i, :] = mean_squared_error(Y_test_inversed.T, test_output_inversed.T, multioutput = 'raw_values', squared = False)

        # Serialize model to YAML
        model_yaml = model.to_yaml()
        with open("model_param\model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)

        # Serialize weights to HDF5
        model.save_weights("model_param\model.h5")
        print("Saved model to disk")

    # Plot train and validation loss
    print('Average test RMSE: {}'.format(np.mean(rmse[:, j])))
    plt.plot(train, color = 'blue', label = 'train')
    plt.plot(val, color = 'orange', label = 'validation')
    plt.legend()
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('ML_results/' + str(sample_num) + '_' + str(test_num) + '_' + str(feature_num) + 'feat_' + str(unit_num_1) + '_' + str(layer_num[j]) + '_epochs' + str(epochs_num) + \
    '_batch' + str(batch_size_num[j]) + '_rmse_5runs.png')
    plt.show()

sp.io.savemat('ML_results/' + str(sample_num) + '_' + str(test_num) + '_' + str(feature_num) + 'feat_' + str(unit_num_1) + '_' + str(layer_num[j]) + '_epochs' + str(epochs_num) + \
'_batch' + str(batch_size_num[j]) + '_rmse_5runs.mat', dict(rmse = rmse))
sp.io.savemat('ML_results/' + str(sample_num) + '_' + str(test_num) + '_' + str(feature_num) + 'feat_' + str(unit_num_1) + '_' + str(layer_num[j]) + '_epochs' + str(epochs_num) + \
'_batch' + str(batch_size_num[j]) + '_rmse_raw_5runs.mat', dict(rmse_raw = rmse_raw))
