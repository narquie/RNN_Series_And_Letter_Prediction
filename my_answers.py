import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0,len(series)-window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (len(X),window_size)
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units = 5,input_shape = (window_size,1),return_sequences=False))
    model.add(Dense(1))
    return(model)




### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    newtext = ""
    for i in text:
        if i in punctuation or i in letters:
            newtext += i
        else:
            newtext += " "
    return newtext

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0,len(text)-window_size,step_size):
        inputs.append(text[i:window_size+i])
        outputs.append(text[window_size + i])
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units = 200,input_shape = (window_size,num_chars),return_sequences=False))
    model.add(Dense(num_chars,activation = 'softmax'))
    return(model)
