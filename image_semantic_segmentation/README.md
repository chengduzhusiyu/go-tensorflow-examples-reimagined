## Image Semantic Segmentation

The example uses DeepLab [mobilenetv2_coco_voc_trainaug](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz) trained on PASCAL VOC 2012.

Refer to [TensorFlow DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) and [DeepLab: Deep Labelling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab) for more details.

### The input and output nodes of the model

| Node Name            | Input/Output | Shape                     | Data Description                                             |
| -------------------- | ------------ | ------------------------- | ------------------------------------------------------------ |
| ImageTensor          | Input        