# Inception-ResNet-TF2

It is the realization of Inception ResNet v2 in both Keras 2.4.3ensorflow 2.2. It has been 
built with the basic modules of both Keras and tensorflow.keras. Accoding to the changed 
scenarios, it includes two basic variants. The first variant almost completely complies with 
the official Inception ResNet paper. And the second model is based on keras' published model
"inception_resnet_v2" and the slim model by TensorFlow. 

The official slim model is hard to be understood due to redundant lines of code and lack of 
clarity. Furthermore, the model in Keas library also inlcudes puzzling code(with the 5x5 filter
in the inception_a). And the last is that the exlicit bug is obvious. Therefore, I greatly 
changes the lines of code to reflect the simplicity and correctedness. 

It is worth noting that the orginal keras and slim models have a different stem. Taht is quite 
different from the published paper, Inception-v4, Inception-ResNet and the Impact of Residual 
Connections on Learning. Furthermore, the lambda function in the above two models drives the 
total size of parameters up 37.5%. It is incurred by the lambda function that generats a serious
side effect. 

The Inception-Resnet A, B and C blocks are 35 x 35, 17 x 17 and 8 x 8 respectively in the grid 
size. Please note the filters in the joint convoluation for A, B and C blocks are respectively 
384, 1154 and 2048. However, the realistic scenario is a little different. 
