#!/usr/bin/env python
# coding: utf-8

# inception_resnet_v2_heavy.py

"""
The Inception-Resnet A, B and C blocks are 35 x 35, 17 x 17 and 8 x 8 respectively in the gride size. 
Please note the filters in the joint convoluation for A B and C blocks are respectively 384, 1154 and 
2048. After adopting the folowing lambda fucntion, the originally filters (384, 1154, 2048) are changed
to (384, 1152, 2144). Please run the command as follows. 

$ python inception_resnet_v2_heavy.py

The total size of parameters of the current model is 58+ million(similar with the official Slim and Keras
model) due to adopting the heavyweight lambda function that addresses the composite operation between the 
input and the residual with computing the operand of the muliplication and then execuing addition. It is 
similar to both the official slim and the Keras model. 

    if scale: e = Lambda(lambda inputs: inputs[0] + inputs[1]*0.17)([input, e])

In contrast, the lightweight lambda addresses the multiplying computation, and then it execute the typical 
concatenation operation. So the lightweight model decreases 45% of the total size of total parameters.  

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code.  

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57.

Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
https://arxiv.org/pdf/1602.07261.pdf 
"""


import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Lambda, Flatten, Activation, \
    BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def inception_stem(input):
    # Define the stem network as similar as Inception v4. 
    a = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid')(input)
    a = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(a)
    a = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(a)

    b1 = Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid')(a)
    b2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(a)

    b = concatenate([b1, b2], axis=3)

    c1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same')(b)
    c1 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(c1)

    c2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same')(b)
    c2 = Conv2D(filters=64, kernel_size=(7,1), strides=(1,1), activation='relu', padding='same')(c2)
    c2 = Conv2D(filters=64, kernel_size=(1,7), strides=(1,1), activation='relu', padding='same')(c2)
    c2 = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(c2)

    c = concatenate([c1, c2], axis=3)

    d1 = Conv2D(filters=192, kernel_size=(3,3), activation='relu', strides=(2,2), padding='valid')(c)
    d2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(c)

    d = concatenate([d1, d2], axis=3)
    d = BatchNormalization(axis=3)(d)
    d = Activation('relu')(d)

    return d


def inception_resnet_a(input, scale):
    # Define Inception-Resnet-A with Inception v4: 10 iterations 
    e1 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)

    e2 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)
    e2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(e2)

    e3 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(input)
    e3 = Conv2D(filters=48, kernel_size=(3,3), activation='relu', padding='same')(e3)
    e3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(e3)

    e = concatenate([e1, e2, e3], axis=3)

    e = Conv2D(filters=384, kernel_size=(1,1), activation='linear', padding='same')(e)
    if scale: e = Lambda(lambda inputs: inputs[0] + inputs[1]*0.17)([input, e])
    e = BatchNormalization(axis=3)(e)
    e = Activation("relu")(e)

    return e


def reduction_a(input): 
    # Define Reduction-A: 35 x 35 --> 17 x 17 
    f1 = Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid')(input)

    f2 = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding='same')(input)
    f2 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(f2)
    f2 = Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid')(f2)

    f3 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    f = concatenate([f1, f2, f3], axis=3)

    return f


def inception_resnet_b(input, scale):
    # Define Inception-Resnet-B: 20 iterations 
    g1 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)

    g2 = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same')(input)
    g2 = Conv2D(filters=160, kernel_size=(1,7), activation='relu', padding='same')(g2)
    g2 = Conv2D(filters=192, kernel_size=(7,1), activation='relu', padding='same')(g2)

    g =concatenate([g1, g2], axis=3)

    # The current number of filters is 1152 rather than 1154. 
    g = Conv2D(filters=1152, kernel_size=(1,1), activation='linear', padding='same')(g) 
    if scale: g = Lambda(lambda inputs: inputs[0] + inputs[1]*0.10)([input, g])
    g = BatchNormalization(axis=3)(g)
    g = Activation("relu")(g)

    return g


def reduction_b(input):
    # Define Reduction-A: 17 x 17 --> 8 x 8  
    h1 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h1 = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(2,2), padding='valid')(h1)

    h2 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h2 = Conv2D(filters=288, kernel_size=(3,3), activation='relu', strides=(2,2), padding='valid')(h2)

    h3 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same')(input)
    h3 = Conv2D(filters=288, kernel_size=(3,3), activation='relu', padding='same')(h3)
    h3 = Conv2D(filters=320, kernel_size=(3,3), activation='relu', strides=(2,2), padding='valid')(h3)

    h4 = MaxPooling2D((3,3), strides=(2,2),padding='valid')(input)

    h = concatenate([h1, h2, h3, h4], axis=3)
    h = BatchNormalization(axis=3)(h)
    h = Activation('relu')(h)

    return h


def inception_resnet_c(input, scale):
    # Define Inception-Resnet-C: 10 iterations 
    i1 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)

    i2 = Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same')(input)
    i2 = Conv2D(filters=224, kernel_size=(1,3), activation='relu', padding='same')(i2)
    i2 = Conv2D(filters=256, kernel_size=(3,1), activation='relu', padding='same')(i2)

    i = concatenate([i1, i2], axis=3)

    # The current number of filters is 2144 rather than 2048. 
    i = Conv2D(filters=2144, kernel_size=(1,1), activation='linear', padding='same')(i) 
    if scale: i = Lambda(lambda inputs: inputs[0] + inputs[1]*0.20)([input, i])
    i = BatchNormalization(axis=3)(i)
    i = Activation("relu")(i)

    return i


def inception_resnet_v2(input_shape, num_classes, include_top, weights):
    # Build the abstract Inception v4 network
    """
    Args:
    input_shape: three dimensions in the TensorFlow Data Format
    num_classes: number of classes
    weights: pre-defined Inception v4 weights 
    include_top: a boolean, for full traning or finetune 

    Return: 
    logits: the logit outputs of the model.
    """
    # Initizate a 3D shape(weight,height,channels) into a 4D tensor(batch, weight, 
    # height, channels). If no batch size, it is defaulted as None.
    inputs = Input(shape=input_shape)

    x = inception_stem(inputs)

    # 10 x Inception Resnet A
    for i in range(10):
        x = inception_resnet_a(x, scale=True)

    # Reduction A - 35 x 35 --> 17 x 17 
    x = reduction_a(x)

    # 20 x Inception Resnet B
    for i in range(20):
        x = inception_resnet_b(x, scale=True)

    # Reduction B - 17 x 17 --> 8 x 8
    x = reduction_b(x)

    # 10 x Inception Resnet C
    for i in range(10):
        x = inception_resnet_c(x, scale=True)

    # Final pooling and prediction
    if include_top:
        x = GlobalAveragePooling2D(name='main_gav_pool')(x)
        x = Dense(units=1536, activation='relu')(x) 
        x = Dropout(0.20)(x)
        x = Dense(units=num_classes, activation='softmax', name='main_pred')(x)

    # Build the model 
    model = Model(inputs, x, name='Inception-Resnet-v2')

    return model


if __name__ == '__main__':

    input_shape = (299,299,3)
    num_classes = 1000
    include_top = True 
    weights = None 

    model = inception_resnet_v2(input_shape, num_classes, include_top, weights)
    model.summary()
