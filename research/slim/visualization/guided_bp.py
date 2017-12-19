import os
import sys

import urllib
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))

from applications import APPBaseModel, slim, tf
from tensorflow.python.framework import ops

os.environ["CUDA_VISIBLE_DEVICES"]=""

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
  dtype = op.outputs[0].dtype
  gate_g = tf.cast(grad > 0, dtype)
  gate_y = tf.cast(op.inputs[0] > 0, dtype)
  return grad * gate_g * gate_y


num_classes = 3780
g = tf.get_default_graph()
with g.gradient_override_map({'Relu': 'GuidedRelu'}):
  m = APPBaseModel('resnet_v1_50', '../application/model.ckpt-538531', num_classes, 
            prepare_image_size=256, 
            outputs=['predictions'],
            label_file='/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_data_v2/label.txt')

img_src = 'http://img08.oneniceapp.com/upload/show/2017/04/16/4ed1a297c1b235c2a9ec65fb177250c5-show.n640.jpg'
image = urllib.urlopen(img_src).read()

label = tf.argmax(m._layers['predictions'], axis=1)
logits = m._layers['resnet_v1_50/spatial_squeeze']
labels = slim.one_hot_encoding([1380], num_classes)
loss = tf.reduce_sum(tf.multiply(logits, labels))

dest_layer_name = 'processed_images'
dest_layer = m.get_all_layers()[dest_layer_name]
grad = tf.gradients(loss, dest_layer)[0]
#norm_grad = tf.div(grad, tf.sqrt(tf.reduce_mean(tf.square(grad))) + tf.constant(1e-5))
norm_grad = grad
[grad_output, label_val] = m._sess.run([norm_grad, label], feed_dict={m._placeholder:[image]})
print (grad_output.shape)
label_val = label_val[0]
print("%d:%s" % (label_val, m._label2text[label_val]))
grad_output = grad_output[0,:,:,:]
cam = grad_output
print(cam)

#cam -= np.min(cam)
cam = np.maximum(cam, 0)
cam /= np.max(cam)
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
#print(heatmap)
heatmap[np.where(heatmap < 0.2*255)] = 0

crop_imgs = m.get_crop_images([image])[0]
print (crop_imgs.shape)

img = heatmap #*0.5 + crop_imgs[:,:,::-1]
cv2.imwrite('3.jpg', img)