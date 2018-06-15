# -*- coding: utf-8 -*-
"""
Created on Sat May 19 05:58:53 2018

@author: Helat
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tensorflow DNNRegressor in Python
# CC-BY-2.0 Paul Balzer
# see: http://www.cbcity.de/deep-learning-tensorflow-dnnregressor-einfach-erklaert
#
TRAINING = False
WITHPLOT = False

# Import Stuff
import tensorflow.contrib.learn as skflow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
np.set_printoptions(precision=2)

#from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__) # 1.4.1

# This is the magic function which the Deep Neural Network
# has to 'learn' (see http://neuralnetworksanddeeplearning.com/chap4.html)
#f = lambda x: 0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)

# Generate the 'features'
#X = np.linspace(0, 1, 1000001).reshape((-1, 1))

# Generate the 'labels'
#y = f(X)



X_train = np.loadtxt('X_train.csv', delimiter=',')
X_dev = np.loadtxt('X_dev.csv',delimiter=',')
X_test = np.loadtxt('X_test.csv',delimiter=',')
y_train = np.loadtxt('Y_train.csv', delimiter=',')
Y_dev = np.loadtxt('Y_dev.csv', delimiter=',')
y_test = np.loadtxt('Y_test.csv', delimiter=',')


# In[3]:





# In[4]:


print ("number of training examples = " + str(X_train.shape[1]))
print ("number of dev examples = " + str(X_dev.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("Y_dev shape: " + str(Y_dev.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

X_train=X_train.T

print ("X_train shape:after transpose " + str(X_train.shape))
X_test=X_test.T

print ("X_test shape: after trans" + str(X_test.shape))

# Defining the Tensorflow input functions
# for training
def training_input_fn():
	return tf.estimator.inputs.numpy_input_fn(
					x={'X': X_train.astype(np.float32)},
					y=y_train.astype(np.float32),
					shuffle=True)
# for test
def test_input_fn():
	return tf.estimator.inputs.numpy_input_fn(
				  	x={'X': X_test.astype(np.float32)},
				  	y=y_test.astype(np.float32),
				  	num_epochs=1,
				  	shuffle=False)

# Network Design
# --------------
feature_columns = [tf.feature_column.numeric_column('X', shape=(1,995))]

STEPS_PER_EPOCH = 100
EPOCHS = 500
BATCH_SIZE = 100

hidden_layers = [16, 16, 16, 16, 16]
dropout = 0.1



#logging.info('Saving to %s' % MODEL_PATH)

# Validation and Test Configuration
validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}

#test_config = skflow.RunConfig(save_checkpoints_steps=100,
#				save_checkpoints_secs=None)

# Building the Network
regressor = skflow.LinearRegressor(feature_columns=feature_columns,
				label_dimension=1,
				#hidden_units=hidden_layers,
				
				#dropout=dropout,
				)

# Train it



		# Fit the DNNRegressor (This is where the magic happens!!!)
regressor.fit(input_fn=training_input_fn())
		# Thats it -----------------------------
		# Start Tensorboard in Terminal:
		# 	tensorboard --logdir='./DNNRegressors/'
		# Now open Browser and visit localhost:6006\

		
		# This is just for fun and educational purpose:
		# Evaluate the DNNRegressor every 10th epoch
	
eval_dict = regressor.evaluate(input_fn=test_input_fn())
y_pred_train = regressor.predict(x={'X': X_train}, as_iterable=False)

y_pred_test = regressor.predict(x={'X': X_test}, as_iterable=False)
Err = (y_train.reshape((1,-1))-y_pred_train)
MSErr = np.mean(Err**2.0)
print('MSE %.5f', eval_dict)
			
				
#plt.scatter(X_test, y_test, label='Real Data')
plt.plot(X_test, y_pred_test, 'r', label='Predicted Data')
plt.xlabel('Neural Network') 
plt.ylabel('Artificial Intelligence')
plt.legend();
fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(1, 4, (1, 3))
ax1.plot(X_train, y_train, label='function to predict')
ax1.plot(X_train, y_pred_train, label='DNNRegressor prediction')
ax1.legend(loc=2)
ax1.set_title("hi")
ax1.set_ylim([0, 1])
                               
ax2 = fig.add_subplot(1, 4, 4)
ax2.set_ylabel('Mean Square Error')
ax2.set_ylim([0, 0.01])
                       
				
			


	# Now it's trained. We can try to predict some values.
logging.info('No training today, just prediction')
np.savetxt("compoundstested.csv",X_test)
np.savetxt("hpredictionsLRsecond.csv",y_pred_test)  