#specific to extracting information from word documents
import pathlib
from get_docx_text import get_docx_text

#converting letters to numbers
import numpy as np

#deep learning modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#alternative to print
import sys

#load files and extract text
FILE_PATH = r'C:\Users\bless\OneDrive\Documents\Goals Docs\Reading docs'
source = pathlib.Path(FILE_PATH)
files = [s for s in source.iterdir()]
texts = [get_docx_text(file).lower() for file in files]
train_text = ' '.join(texts)

#create numberical map using document characters
chars = sorted(list(set(train_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(train_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = train_text[i:i + seq_length]
	seq_out = train_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
    
n_patterns = len(dataX)


# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.load_weights('control-starting-weight.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

#define checkpoint
filepath="custom-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#-----------------------------------------------COMMENT OUT IF NOT GENERATING WEIGHTS---------------------------------------------------------------------------
#fit model
#model.fit(X, y, epochs=2, batch_size=60, callbacks=callbacks_list)
#-----------------------------------------------COMMENT OUT IF NOT GENERATING WEIGHTS---------------------------------------------------------------------------

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for i in range(300):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")





























