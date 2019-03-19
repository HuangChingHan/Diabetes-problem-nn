# -*- coding: utf-8 -*-
"""
Date: 2019/03/19

This is the first keras network.

"""
# Part 1: Load Data
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Part 2: Define Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Part 3: Compile Model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# Part 4: Fit Model
model.fit(X,Y,epochs=150,batch_size=10)
'''
# Part 5: Evaluate Model
scores = model.evaluate(X,Y)
print("\n%s : %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''
# Part 6: Make Predictions
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)











