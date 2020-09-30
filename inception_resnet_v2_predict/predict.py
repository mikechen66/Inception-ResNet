#!/usr/bin/env python
# coding: utf-8

# predict.py

"""
Inception-ResNet V2 model for Keras

The Inception-Resnet A, B and C blocks are 35 x 35, 17 x 17 and 8 x 8 respectively in the gride size. 
Please note the filters in the joint convoluation for A B and C blocks are respectively 384, 1154 and 
2048. However, the realistic scenario is a little different. Please run the commands as follows. 

$ python pred.py  

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

Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/slim#pre-trained-models

# Reference
- Inception-v4, Inception-ResNet and the Impact ofResidual Connections on Learning
- https://arxiv.org/abs/1602.07261)
"""

import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from inception_resnet_v2_func import inception_resnet_v2
from keras.applications.imagenet_utils import decode_predictions


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x):
    
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output 


if __name__ == '__main__':

    input_shape = (229,299,3)

    model = inception_resnet_v2(input_shape)
    model.summary()

    img_path = '/home/mike/Documents/keras_inception_resnet_v2/elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    output = preprocess_input(img)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))