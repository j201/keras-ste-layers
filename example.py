import keras
import keras.optimizers as optimizers
from keras.datasets import cifar10
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, AveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np
from ste import STE

# Load and preprocess data
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
m = np.mean(x_train)
s = np.std(x_train)
x_train = (x_train-m)/(s + 1e-7)
x_test = (x_test-m)/(s + 1e-7)

# Build LeNet-5 model
use_ste_layers = True # Flip to test with normal dense layers instead
ste_dropconnect = False # Flip to use dropconnect in STE layers rather than dropout
ensemble_size = 8
input_layer = Input(shape=input_shape)
x = Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), padding='same', activation='tanh')(input_layer)
x = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(x)
x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), padding='same', activation='tanh')(x)
x = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(x)
x = Flatten()(x)
for n in [120,84]:
	if use_ste_layers:
		x = STE(n, ensemble_size=ensemble_size, activation='tanh', dropconnect=ste_dropconnect)(x)
	else:
		x = Dense(n, activation='tanh')(x)
	x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(input=input_layer, output=x)

# Train model
batch_size = 128
n_epochs = 256
opt = optimizers.SGD(lr=1e-2, decay=3e-4, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
			  optimizer=opt,
			  metrics=['accuracy'])
keras.utils.print_summary(model)
es = EarlyStopping(patience=8, restore_best_weights=True)
model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=n_epochs, verbose=1, callbacks=[es])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
