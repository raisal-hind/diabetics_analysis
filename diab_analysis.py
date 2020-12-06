from keras.models import Sequential
from keras.layers import Dense
import numpy

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
print(dataset.shape)
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
print(X.shape)
print(Y.shape)

# Extending the first model with activation functions
model = Sequential()
#specifying activation functions
model.add(Dense(3, input_dim=8, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=20, batch_size=10,  verbose=2)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()

for layer in model.layers:
    print(layer.name, layer.inbound_nodes, layer.outbound_nodes)
    
    
model.get_weights()

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(predictions)
print(rounded)

