# Deep Learning Blogs

A curated list of machine Leaning blog posts 


## General Concepts

### Convolution

* [2D , 3D , 1x1 , Transposed , Dilated (Atrous) , Spatially Separable , Depthwise Separable , Flattened , Grouped , Shuffled Grouped Convolutions](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)  
* [Max Pooling/Average Pooling/Global Average Pooling](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) 

### Optimization

* [RMSprop,RProp,Adagrad](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)
* [Batch,Stochastic,Mini-batch gradient descent,Momentum,Nesterov accelerated gradient,Adagrad,Adadelta,RMSprop,Adam,AdaMax,Nadam,AMSGrad](https://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient)
* [Stochastic,Mini-batch gradient descent,Momentum,Nesterov accelerated gradient,RProp,RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) 

### Activation Functions

* [ReLu,Leaky ReLu,PRelu,Elu,Selu,CRelu,Relu-6](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7)
* [GELU, SELU, ELU, ReLU](https://mlfromscratch.com/activation-functions-explained/)
* [Sigmoid, ReLU, LReLU, PReLU, RReLU, ELU, Softmax](http://laid.delanover.com/activation-functions-in-deep-learning-sigmoid-relu-lrelu-prelu-rrelu-elu-softmax/)

### Loss Functions
* [MSE,MAE,Huber Loss,Cross Entropy, Hinge Loss, Softmax,  KL-Divergence](https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/)
* [Kullback-Leibler Divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) 
* [Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

<a name="github-tutorials" />

### Regularization

* [Batch normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) 
* [Label Smoothing](https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0) 
* [Dropout Math](https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275)
## Image Classification

### Residual Networks

* [ResNet,ResNext,Deep Network with Stochastic Depth,Densenet,ResNet as an Ensemble of Smaller Networks](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
* [Preactivation Resnet](https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e)
* [Highway vs Residual Networks?](https://www.quora.com/What-are-the-differences-between-Highway-Networks-and-Deep-Residual-Learning)
* [Inverted Residuals and Linear Bottlenecks](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)
* [DenseNets](https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a)
* [Squeeze-and-Excitation Networks](https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7)

### Inception Networks

* [Inception v1/v2/v3/v4/Resnet](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202), split-transform-merge paradighm , auxiliary classifiers, 
* [Xeption](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568),Depthwise Separable Convolution
* [SqueezeNet](https://towardsdatascience.com/review-squeezenet-image-classification-e7414825581a)


## Object Detection


### Overfeat
* [Overfeat](https://towardsdatascience.com/object-localization-in-overfeat-5bb2f7328b62)

### Yolo
* [YOLO v1,v2,v3](https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088), [Non-maximum Suppression](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c), [Anchor Boxes](https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html)
* [YOLO v1](https://hackernoon.com/understanding-yolo-f5a74bbc7967)
* [YOLO v2](https://medium.com/@y1017c121y/how-does-yolov2-work-daaaa967c5f7)
* [YOLO v3](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

### RCNN
* [Faster R-CNN, R-FCN, FPN](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9)
* [RCNN, Fast RCNN, Faster RCNN](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)
* [Spatial Pyramid Pooling](https://medium.com/coinmonks/review-sppnet-1st-runner-up-object-detection-2nd-runner-up-image-classification-in-ilsvrc-906da3753679)
* [Fast RCNN](https://towardsdatascience.com/fast-r-cnn-for-object-detection-a-technical-summary-a0ff94faa022), [ROI Pooling](https://deepsense.ai/region-of-interest-pooling-explained/)
* [RCNN, Fast RCNN, SPP-Net,Faster RCNN](https://slideplayer.com/slide/13427815/)
* [Faster RCNN](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8), [Region Purpose Network](https://www.quora.com/How-does-the-region-proposal-network-RPN-in-Faster-R-CNN-work)
* [Mask RCNN ](https://medium.com/@tibastar/mask-r-cnn-d69aa596761f ), [Details](https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296), 
* [RFCN](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99)

### SSD
* [SSD, YOLO, FPN, Retina Net](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)
* [SSD](https://medium.com/inveterate-learner/real-time-object-detection-part-1-understanding-ssd-65797a5e675b)
* [Retina Net](https://towardsdatascience.com/retinanet-how-focal-loss-fixes-single-shot-detection-cb320e3bb0de), Focal loss

### UNet
* [UNet](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47), [Upsampling with transposed convolution](https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0)


## Attention
* [Attention](https://towardsdatascience.com/visual-attention-model-in-deep-learning-708813c2912c), 


## Super Resolution
* [Survey](https://medium.com/beyondminds/an-introduction-to-super-resolution-using-deep-learning-f60aff9a499d)
