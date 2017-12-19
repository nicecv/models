import os
import sys

import urllib
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))

from applications import APPBaseModel, slim, tf

os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes = 3780
m = APPBaseModel('resnet_v1_50', '../application/model.ckpt-538531', num_classes, 
          prepare_image_size=256, 
          outputs=['predictions'],
          label_file='/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_data_v2/label.txt')

img_src = 'http://img08.oneniceapp.com/upload/show/2017/12/11/30cf30c9b80b8cdf8087a3ef95437c1f-show.n640.jpg'
image = urllib.urlopen(img_src).read()
with open('a.jpg', 'w') as fw:
  fw.write(image)
features = m.predict([image])

predictions = features['predictions']
label = np.argmax(predictions)
sorted_label = np.argsort(-predictions)[0]
print (sorted_label)
for i in xrange(10):
  print("%d:%s" % (sorted_label[i], m._label2text[sorted_label[i]]))

logits = m._layers['resnet_v1_50/spatial_squeeze']
labels = slim.one_hot_encoding([sorted_label[7]], num_classes)
loss = tf.reduce_sum(tf.multiply(logits, labels))

#print (m.get_all_variables())
#print (m.get_all_layers())
dest_layer_name = 'resnet_v1_50/block4'
dest_layer = m.get_all_layers()[dest_layer_name]
grad = tf.gradients(loss, dest_layer)[0]
norm_grad = tf.div(grad, tf.sqrt(tf.reduce_mean(tf.square(grad))) + tf.constant(1e-5))

[conv_output, grad_output] = m._sess.run([dest_layer, norm_grad], feed_dict={m._placeholder:[image]})
conv_output = conv_output[0,:,:,:]
grad_output = grad_output[0,:,:,:]
weights = np.mean(grad_output, axis=(0,1))

cam = np.zeros(dtype = np.float32, shape = conv_output.shape[:2])
for i, w in enumerate(weights):
  cam += w * conv_output[:, :, i]
    
#cam = cam * (cam>0)
cam = np.maximum(cam, 0)
cam /= np.max(cam)
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0

crop_imgs = m.get_crop_images([image])[0]
print (crop_imgs.shape)

img = heatmap *0.5 + crop_imgs[:,:,::-1]

cv2.imwrite('2.jpg', img)