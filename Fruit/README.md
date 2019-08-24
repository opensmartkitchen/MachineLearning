# MachineLearning: Fruit360 Classifier

A reimplementation of [Fruit-360 CNN TensorFlow](https://www.kaggle.com/mitch9090/fruit-360-cnn-tensorflow "Fruit 360").




## Get Started

Download the Fruits360 dataset. We used version Version 44 of 60, but we will eventually update to the latest version of the dataset which can be found [here](https://www.kaggle.com/moltean/fruits). Put the folder named fruits-360 inside the folder fruits-360.

Tensorflow 2 was used, but due to stability issues, we disabled the 2.0 behavior.

```python
# python 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
If you use a version below 2.0, use 
```python 
import tensorflow as tf
```
