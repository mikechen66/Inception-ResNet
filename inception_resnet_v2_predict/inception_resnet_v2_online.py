#!/usr/bin/env python
# coding: utf-8

# inception_resnet_v2_online.py

"""
Inception-ResNet V2 model for Keras

The Inception-Resnet A, B and C blocks are 35 x 35, 17 x 17 and 8 x 8 respectively in the gride size. 
Please note the filters in the joint convoluation for A B and C blocks are respectively 384, 1154 and 
2048. However, the realistic scenario is a little different. Please run the commands as follows. 

$ python inception_resnet_v2_online.py

The total size of parameters of the current model is 55+ million(similar to the official Slim model) 
due to adopting the heavy weight lambda function that addresses the composite operation between the 
input and the residual with computing the operand of the muliplication and then execuing addition. 

    mix = Lambda(lambda inputs, scale: inputs[0]+inputs[1]*scale, 
                 output_shape=K.int_shape(input)[1:], 
                 arguments={'scale': scale}, 
                 name=block_name)([input, mix])

In contrast, the lightweight lambda(inception_resnet_v2_tf2)addresses the multiplying computation, and
then it execute the typical concatenation operation. So the lightweight model decreases 35%~45% of the 
total size of total parameters.  

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

Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/slim#pre-trained-models

# Reference
- Inception-v4, Inception-ResNet and the Impact ofResidual Connections on Learning
- https://arxiv.org/abs/1602.07261)
"""


import warnings
import numpy as np
import tensorflow as tf 
from keras.models import Model

from keras import backend as K
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, Input, Lambda, Activation, Concatenate, BatchNormalization, \
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.utils.data_utils import get_file
from imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import decode_predictions


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Assume users have already downloaded the Inception v4 weights 
BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', 
              use_bias=False, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, 
               use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)

    return x


def inception_stem(input):
    # Stem block: 35 x 35 x 192
    x = conv2d_bn(input, filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')
    x = conv2d_bn(x, filters=32, kernel_size=(3,3), padding='valid')
    x = conv2d_bn(x, filters=64, kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = conv2d_bn(x, filters=80, kernel_size=(1,1), padding='valid')
    x = conv2d_bn(x, filters=192, kernel_size=(3,3), padding='valid')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    return x 


def inception_a(input):
    # Inception-A block: 35 x 35 x 320
    branch_11 = conv2d_bn(input, filters=96, kernel_size=(1,1))

    branch_12 = conv2d_bn(input, filters=48, kernel_size=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=64, kernel_size=(5,5))

    branch_13 = conv2d_bn(input, filters=64, kernel_size=(1,1))
    branch_23 = conv2d_bn(branch_13, filters=96, kernel_size=(3,3))
    branch_33 = conv2d_bn(branch_23, filters=96, kernel_size=(3,3))

    branch_14 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    branch_24 = conv2d_bn(branch_14, filters=64, kernel_size=(1,1))

    branches = [branch_11, branch_22, branch_33, branch_24]

    x = Concatenate(axis=3, name='mixed_5b')(branches)

    return x 


def inception_resnet_block(input, scale, block_type, block_idx, activation='relu'):
    # Add an Inception-ResNet block.
    if block_type == 'block35':

        branch_11 = conv2d_bn(input, filters=32, kernel_size=(1,1))

        branch_12 = conv2d_bn(input, filters=32, kernel_size=(1,1))
        branch_22 = conv2d_bn(branch_12, filters=32, kernel_size=(3,3))

        branch_13 = conv2d_bn(input, filters=32, kernel_size=(1,1))
        branch_23 = conv2d_bn(branch_13, filters=48, kernel_size=(3,3))
        branch_33 = conv2d_bn(branch_23, filters=64, kernel_size=(3,3))

        branches = [branch_11, branch_22, branch_33]

    elif block_type == 'block17':

        branch_11 = conv2d_bn(input, filters=192, kernel_size=(1,1))

        branch_12 = conv2d_bn(input, filters=128, kernel_size=(1,1))
        branch_22 = conv2d_bn(branch_12, filters=160, kernel_size=(1,7))
        branch_32 = conv2d_bn(branch_22, filters=192, kernel_size=(7,1))

        branches = [branch_11, branch_32]

    elif block_type == 'block8':

        branch_11 = conv2d_bn(input, filters=192, kernel_size=(1,1))

        branch_12 = conv2d_bn(input, filters=192, kernel_size=(1,1))
        branch_22 = conv2d_bn(branch_12, filters=224, kernel_size=(1,3))
        branch_32 = conv2d_bn(branch_22, filters=256, kernel_size=(3,1))

        branches = [branch_11, branch_32]

    else:

        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)

    concat = Concatenate(axis=3, name=block_name + '_mixed')(branches)
    mix = conv2d_bn(concat, K.int_shape(input)[3], kernel_size=(1,1), activation=None, 
                   use_bias=True, name=block_name + '_conv')
    # If we divide the addition of between the input and residual(the operand of the 
    # muliplication, tptal size of parameters will decrease 45%. 
    mix = Lambda(lambda inputs, scale: inputs[0]+inputs[1]*scale, 
                 output_shape=K.int_shape(input)[1:], 
                 arguments={'scale': scale}, 
                 name=block_name)([input, mix])

    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(mix)

    return x


def reduction_a(input):
    # Mixed 6a (Reduction-A block): 17 x 17
    branch_11 = conv2d_bn(input, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv2d_bn(input, filters=256, kernel_size=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=256, kernel_size=(3,3))
    branch_32 = conv2d_bn(branch_22, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(input)

    branches = [branch_11, branch_32, branch_13]

    x = Concatenate(axis=3, name='mixed_6a')(branches)

    return x 


def reduction_b(input):
    # Mixed 7a (Reduction-B block): 8 x 8 
    branch_11 = conv2d_bn(input, filters=256, kernel_size=(1,1))
    branch_21 = conv2d_bn(branch_11, filters=384, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_12 = conv2d_bn(input, filters=256, kernel_size=(1,1))
    branch_22 = conv2d_bn(branch_12, filters=288, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_13 = conv2d_bn(input, filters=256, kernel_size=(1,1))
    branch_23 = conv2d_bn(branch_13, filters=288, kernel_size=(3,3))
    branch_33 = conv2d_bn(branch_23, filters=320, kernel_size=(3,3), strides=(2,2), padding='valid')

    branch_14 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(input)

    branches = [branch_21, branch_22, branch_33, branch_14]

    x = Concatenate(axis=3, name='mixed_7a')(branches)

    return x 


def InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None,
                      input_shape=None, pooling=None, classes=1000):
    # Determine proper input shape (-K.image_data_format())
    input_shape = _obtain_input_shape(input_shape, default_size=299, min_size=139, 
                                      data_format=None, weights=weights,
                                      require_flatten=include_top)

    # Initizate a 3D shape into a 4D tensor with a batch. If no batch size, 
    # it is defaulted as None.
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Call the function of inception_stem()
    x = inception_stem(inputs)

    # Call the function of inception_a
    x = inception_a(x)

    # 10 x Inception-ResNet-A block: 35 x 35 x 320
    for block_idx in range(0, 10):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)

    # Reduction-A Block 
    x = reduction_a(x)

    # 20 x Inception-ResNet-B block: 17 x 17 x 1088
    for block_idx in range(0, 20):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)

    # Reduction-B Block 
    x = reduction_b(x)

    # 10 x Inception-ResNet-C block: 8 x 8 x 2080
    for block_idx in range(0, 10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v2')

    # Load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_last':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, the setting'
                              'is `image_data_format="channels_last"` in the keras'
                              'config at ~/.keras/keras.json.')
        if include_top:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)

    return model


def preprocess_input(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output 


if __name__ == '__main__':

    input_shape = (229,299,3)

    model = InceptionResNetV2(input_shape)

    model.summary()

    img_path = '/home/mike/Documents/keras_inception_resnet_v2/elephant.jpg'
    img = image.load_img(img_path, target_size=(299,299))
    output = preprocess_input(img)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))
