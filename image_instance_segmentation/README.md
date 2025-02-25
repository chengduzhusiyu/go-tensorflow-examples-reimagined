
## Image Instance Segmentation

The example uses [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) trained on the COCO dataset.

Refer to [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md) for more details.

### The input and output nodes of the model

| Node Name         | Input/Output | Shape                             | Data Description                                                               |
| ----------------- | ------------ | --------------------------------- | ------------------------------------------------------------------------------ |
| image_tensor      | Input        | [batch, height, width, 3]         | RGB pixel values as uint8 in a square format (Width, Height).                  |
| detection_boxes   | Output       | [batch, num_detections, 4]        | Array of boxes for each detected object in the format [yMin, xMin, yMax, xMax] |
| detection_scores  | Output       | [batch, num_detections]           | Array of probability scores for each detected object between 0 and 1           |
| detection_classes | Output       | [batch, num_detections]           | Array of object class indices for each object detected based on COCO objects   |
| detection_masks   | Output       | [batch, mask_height, mask_height] | Array of instance masks                                                        |

### Usage

`go run main.go -dir=<model folder> -jpg=<input.jpg> [-out=<output.jpg>] [-labels=<labels.txt>]`

### References

- [Run an Instance Segmentation Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md)
- TensorFlow [visualization_utils](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py)