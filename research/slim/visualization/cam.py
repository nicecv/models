import os
import sys

import urllib
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))

from applications import APPBaseModel

os.environ["CUDA_VISIBLE_DEVICES"]=""

m = APPBaseModel('resnet_v1_50', '../application/model.ckpt-538531', 3780, 
          prepare_image_size=256, 
          outputs=['predictions', 'resnet_v1_50/block4'],
          label_file='/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_data_v2/label.txt')

class_weights = m.get_weights('resnet_v1_50/logits/weights:0')[0,0,:,:]

img_src = 'http://img08.oneniceapp.com/upload/show/2017/08/29/47bf8adf7d34a7b9c07051a8f43f6587-show.n640.jpg'
image = urllib.urlopen(img_src).read()
features = m.predict([image])
conv_output = features['resnet_v1_50/block4']
conv_output = conv_output[0,:,:,:]

predictions = features['predictions']
label = np.argmax(predictions)

print("%d:%s" % (label, m._label2text[label]))

cam = np.zeros(dtype = np.float32, shape = conv_output.shape[:2])
for i, w in enumerate(class_weights[:,label]):
  cam += w * conv_output[:, :, i]

cam -= np.min(cam)
cam /= np.max(cam)
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0

crop_imgs = m.get_crop_images([image])[0]
print (crop_imgs.shape)

img = heatmap *0.5 + crop_imgs[:,:,::-1]
cv2.imwrite('1.jpg', img)