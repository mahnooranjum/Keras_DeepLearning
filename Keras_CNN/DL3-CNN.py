##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#   
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#    Dataset:
#    https://www.kaggle.com/c/dogs-vs-cats
#
##############################################################################


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
'''
Step 1 - Convolution:
    Input Image * Feature Detector 
    Convolution between them gives us a
    quantitative measure of how much the feature
    is present in the input image.
    
    The feature detector slides across the input
    image. The step it takes in one step is called 
    a stride.
    
    The larger the stride, the smaller the output image
    is going to be.
    The output image is called a feature map.
    
    Do we lose information?  YES. But the point of this step is
    to lose useless information and get important features out of
    the input image.
    
    
    We create multiple feature maps using different feature detectors. 
    Feature detectors are basically filters. 
    
    Examples of filters:
        Sharpen
        Blur
        Edge Enhance
        Edge Detect
        Emboss
        
    REVISE:
        WHAT DOES CONVOLUTION DO?
        Gives us useful features.
        
'''
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
'''
    But wait, I understand the convolution part, where did the activation of relu come
    from? 
    
    The purpose of RELU is to increase non linearity. Images are non linear, 
    Our feature maps turn images into linear maps. To obtain non-linearity, we 
    apply ReLU activation. 
    
    While the benefit is hard to comprehend visually, we can imaging a color scale
    from white to black to white. A linear transition will be white, grey, black, grey, white. ReLU turns 
    white to grey to white directly. Hence creating a non linear model.
'''

'''
Step 2 - Pooling:
    How do we recognize a human being if the photo has been taken from a different 
    perspective?
    How do we recognize a person if he is sitting, if he is standing, if the photo
    has been taken from behind the person.
    How do we cater for different perspectives?
    
    We take a box of pixels that slides across the feature map obtained by convolution.
    The box recognizes the MAXIMUM value the pixels have under the box. That maximum value is 
    used to create a POOLED FEATURE MAP and the rest of the pixels are discarded.
    
    Another advantage is the reduction of image size. During the removal of information, we 
    obtain a model that prevents overfitting.
    
    Go to the following link to see CNNs in action.
    
    https://www.cs.cmu.edu/~aharley/vis/

'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''
ADDING ANOTHER LAYER OF CONVOLUTION AND POOLING
'''
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''
Step 3 - Flattening:
    
    Flattening turns our 2D image into a 1D array
    This is one column. Each pixel corresponds to an input
    neuron.
    
'''
classifier.add(Flatten())

'''
Step 4 - Full connection:
    Full connection corresponds to a DENSE ANN.
    We will connect every neuron of one layer, to every neuron of the next layer.
    
    It is the job of the ANN to learn weights and combine neurons to create new attributes.
    We DO NOT CARE about those attributes. We just care about the result.

'''
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

'''
Compiling the CNN:
    The adam optimizer works best for our problem.
    Binary cross entropy loss is the critereon used for a binary 
    classification problem.
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''
Fitting the CNN to the images:
    This code has been taken from the official keras documentation.
    
    It basically transforms input images to create new training images.
'''

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# MAKING PREDICTIONS
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog.jpg',target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'