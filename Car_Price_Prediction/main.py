import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import Normalization, Dense , InputLayer
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError #for error correction -> summation of (pred - actual)square/total elements
from tensorflow.keras.metrics import RootMeanSquaredError
from  tensorflow.keras.optimizers import Adam 


#data preparation
data = pd.read_csv(r"C:\Users\Harish Pavan Rolla\class\deeplearning\carpricepredictor\train.csv")
#print(data.shape)
#b=sns.pairplot(data[["years","km","rating","condition","economy","top speed","hp","torque","current price"]], diag_kind='kde')
#plt.show()
tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data,tf.float32)
tensor_data =tf.random.shuffle(tensor_data)


X = tensor_data[:,3:-1]
y = tensor_data[:,-1] #here actuly u get 1d vector to make it accordingly data use expand_dims
y = tf.expand_dims(y, axis = -1) #shape(1000,1)

#for training and validation and testing
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SEIZE = len(X)

x_train = X[:int(DATASET_SEIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SEIZE*TRAIN_RATIO)]

x_val = X[int(DATASET_SEIZE*TRAIN_RATIO):int(DATASET_SEIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SEIZE*TRAIN_RATIO):int(DATASET_SEIZE*(TRAIN_RATIO+VAL_RATIO))]

x_test = X[int(DATASET_SEIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SEIZE*(TRAIN_RATIO+VAL_RATIO)):]

#using dataset function to optimize the loading and sending data as batches
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)
  
  
#normalization of input => (x-mean)/ standard deveation - sso every value will be between 0 t0 1
#in tensorflow we have tf.keras.layers.Normalization (std= square root of varience)
normalizer = Normalization()
normalizer.adapt(x_train)

#Linear regression model
model = tf.keras.Sequential([
    InputLayer(input_shape = (8,)),# sending 32 at frist like a batch insted of 1000 at a time
    normalizer,
    Dense(128, activation = "relu"),
    Dense(128, activation = "relu"),
    Dense(128, activation = "relu"),
    Dense(128, activation = "relu"),
    Dense(1),#number of outputs
])
#model.summary()

#tf.keras.utils.plot_model(model, to_file="model.png",show_shapes=True)
model.compile(optimizer= Adam(learning_rate=0.1),
              loss = MeanAbsoluteError(),
              metrics=RootMeanSquaredError())


history = model.fit(train_dataset, validation_data=val_dataset ,epochs=100, verbose=1) 


#graph val loss vs training loss(error)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'])
plt.show()

#graph root_mean_err0r vs val_loss
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['root_mean_squared_error','val_loss'])
plt.show()
print(model.evaluate(x_test,y_test))
#print(history.history)

#predict
y_true = list(y_test[:,0].numpy())
y_pred = list(model.predict(x_test)[:,0])

#graph actual vs predict
ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.1

plt.bar(ind, y_pred, width, label='Predicted Car Price')
plt.bar(ind + width, y_true, width, label='Actual Car Price')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car Price Prices')

plt.show()
