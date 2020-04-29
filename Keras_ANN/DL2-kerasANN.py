##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#
#
##############################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# iloc gets data via numerical indexes
# .values converts from python dataframe to numpy object
dataset = pd.read_csv('MoonsANN.csv')
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values

from matplotlib.colors import ListedColormap
for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y == j, 0], X[y == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
plt.clf()




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#Importing library module for splitting dataset into
#test and train subsets
# 15% goes into the test set and the rest goes into the train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

'''
The Sequential model is a linear stack of layers.

You can create a Sequential model by passing a list of layer instances to the constructor:

from keras.models import Sequential

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')])
You can also simply add layers via the .add() method:

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

###########################################################################################
    DENSE:
    Just your regular densely-connected NN layer.
    Dense connections have all neurons connected to each other
    
    Dense implements the operation: output = activation(dot(input, kernel) + bias) 
        activation:
            is the element-wise activation function passed as the activation argument
        kernel:
            is a weights matrix created by the layer
        bias:
            is a bias vector  created by the layer (only applicable if use_bias is True).
            
    ARGUMENTS:
        units: Positive integer, dimensionality of the output space.
        
        activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        
        use_bias: Boolean, whether the layer uses a bias vector.
        
        kernel_initializer: Initializer for the kernel weights matrix (see initializers).
        
        bias_initializer: Initializer for the bias vector (see initializers).
        
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        
        bias_constraint: Constraint function applied to the bias vector (see constraints).
    
'''

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initializing the model
classifier = Sequential()
# Adding the input layer and the first hidden layer
# Input units must equal the dimensions of input variable
# Output units are decided on a test and trial basis
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 20)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Getting the accuracy score
from sklearn.metrics import accuracy_score
ascore = accuracy_score(y_test, y_pred)

# Visualizing
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightsalmon', 'greenyellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN Train Set')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
plt.clf()


# Visualizing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightsalmon', 'greenyellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN Test Set')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()





