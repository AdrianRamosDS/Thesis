# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 06:21:35 2021

@author: Adrian Ramos Perez

"""
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal as sig

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Audio library imports:
import librosa as lb
import librosa.display
import soundfile as sf
from IPython.display import Audio

# Deep Learning imports:   
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
import TDNN_Layer


# def plot_data(pl, x, y):
#     pl.plot(x[y==0, 0], x[y==0, 1], 'ob', alpha=0.5)
#     pl.plot(x[y==1, 0], x[y==1, 1], 'xr', alpha=0.5)
#     pl.legend(['0','1'])
#     return pl

#%% Lectura de archivos de audio 
samples, sampling_rate= lb.load("../Data/ADS.wav", sr=None, mono=True,offset=0.0,duration=None)
# Duración en segundos:
duration = len(samples)/sampling_rate
Audio("../Data/ADS.wav")

ir_samples, ir_sampling_rate= lb.load("../Data/Golpe.wav", sr=None, mono=True,offset=0.0,duration=None)
ir_duration = len(ir_samples)/ir_sampling_rate

#%% Graficar señales de audio
plt.show()
plt.figure()
plt.title('ADS Speech Signal (original)')
lb.display.waveplot(y=samples,sr=sampling_rate)

plt.show()
plt.figure()
plt.title('Impulse Response')
lb.display.waveplot(y=ir_samples,sr=ir_sampling_rate)

# sampling_rate = 44100
# freq = 440
# samples = 44100
# x = np.arange(samples)

#%% Convolution

convolved_signal = np.convolve(samples, ir_samples, mode='same')

plt.show()
plt.figure()
plt.title('Convolved signal')
lb.display.waveplot(y=convolved_signal,sr=sampling_rate)
# signal.astype(np.int16).tofile('../Data/convolved_signal.wav')
sf.write('../Data/convolved_signal.wav',convolved_signal, sampling_rate, subtype=None)

# # Sine wave
# signal = 100*np.sin(2*np.pi*freq*x/sampling_rate)
# plt.show()
# plt.figure()
# plt.title('Senoidal')
# lb.display.waveplot(y=signal,sr=sampling_rate, color='y')
# signal.astype(np.int16).tofile('test.wav')
#%% Delayed Signal

del_sig = samples[:]

#%% Neural Network 
# plt.figure()
# x, y = make_blobs(n_samples = 1000, centers = 2, random_state=42)
# x
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# pl = plot_data(plt, x, y)
# pl.show()

# Model definition:
model = Sequential()

# Layer(s) definition:
model.add(Dense(2, input_shape=(1,), activation="linear", name="input_layer"))
# model.add(Dense(3, activation="tanh", name="hidden_layer"))
model.add(TDNN_Layer())
model.add(Dense(1, activation='linear', name="output_layer"))

# Compile model:
sgd = SGD(learning_rate=0.1, momentum=0.8)
model.compile(optimizer=Adam(lr=0.05), loss='MeanSquaredError')

# Train model:
h = model.fit(convolved_signal, samples, epochs=5, verbose=1)

#%% Model prediction

Pred_data = model.predict(convolved_signal)

plt.show()
plt.figure()
plt.title('Predicted Signal')
lb.display.waveplot(y=Pred_data,sr=sampling_rate)

#%% Evaluate performance:
# eval_result = model.evaluate(x, y)
# print("Error result: loss, accuraccy \n", eval_result)