import os
import numpy as np
import tensorflow as tf

from feature_extractor_util import FeatureExtractionFromFrozenGraph

tf.app.flags.DEFINE_integer(
    'device_id', None, 'The id of device.')

tf.app.flags.DEFINE_string(
    'frozen_model', None, 'The path of frozen model.')

tf.app.flags.DEFINE_string(
    'input_names', 'input:0', 'The names of inputs, split by ","')
        
tf.app.flags.DEFINE_string(
    'output_names', 'output:0', 'The names of outputs, split by ","')

tf.app.flags.DEFINE_string(
    'prefix_name', 'nice', 'The prefixe name of network')

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'batch size')
    
tf.app.flags.DEFINE_string(
    'tf_record_dirname', '', 'The dirname of tensorflow record')
tf.app.flags.DEFINE_string(
    'save_dirname', '', 'The dirname to save result array')

FLAGS = tf.app.flags.FLAGS
    
def main(_):
  feature_extractor_obj = FeatureExtractionFromFrozenGraph(
                            FLAGS.frozen_model, 
                            FLAGS.input_names, 
                            FLAGS.output_names, 
                            FLAGS.prefix_name, 
                            FLAGS.device_id)
  
  save_output_names = [name.replace('/', '+').replace(':', '-') for name in FLAGS.output_names.split(',')]
  results = {name:[] for name in save_output_names}
  for filename in os.listdir(FLAGS.tf_record_dirname):
    print filename
    imgs = []
    for example_serialized in tf.python_io.tf_record_iterator(os.path.join(FLAGS.tf_record_dirname,filename)):
      example = tf.train.Example()
      example.ParseFromString(example_serialized)
      img_raw = example.features.feature['image/encoded'].bytes_list.value[0]
      imgs.append(img_raw)
      if len(imgs) == FLAGS.batch_size:
        outputs = feature_extractor_obj.process([np.array(imgs)])
        imgs = []
        for output,save_output_name in zip(outputs, save_output_names):
          results[save_output_name].append(output.reshape((output.shape[0],-1)))
          
    if len(imgs) > 0:
      outputs = feature_extractor_obj.process([np.array(imgs)])
      imgs = []
      for output,save_output_name in zip(outputs, save_output_names):
        results[save_output_name].append(output.reshape((output.shape[0],-1)))
    for save_output_name in save_output_names:
      np.save(os.path.join(FLAGS.save_dirname, filename+'.'+save_output_name), np.concatenate(results[save_output_name], axis=0))
      results[save_output_name] = []  
      
if __name__ == '__main__':
  tf.app.run()