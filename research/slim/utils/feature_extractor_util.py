import os, sys
import tensorflow as tf

def load_graph(frozen_graph_filename, prefix_name):
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we can use again a convenient built-in function to import a graph_def into the
  # current default Graph
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(
      graph_def,
      input_map=None,
      return_elements=None,
      name=prefix_name,
      op_dict=None,
      producer_op_list=None
    )
  return graph
    
class FeatureExtractionFromFrozenGraph(object):
  def __init__(self, frozen_graph_filename, input_names, output_names, prefix_name, device_id=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    if device_id is not None:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    graph = load_graph(frozen_graph_filename, prefix_name)
    
    self._inputs = [graph.get_tensor_by_name(prefix_name+'/'+input_name) for input_name in 
                     input_names.strip().split(',')]
    self._outputs = [graph.get_tensor_by_name(prefix_name+'/'+output_name) for output_name in
                     output_names.strip().split(',')]
                     
    self._sess = tf.Session(config=config, graph=graph)

  def process(self, input_features):
    input_dict = {}
    for input_feature,input_name in zip(input_features,self._inputs):
      input_dict[input_name] = input_feature
    return self._sess.run(self._outputs, input_dict)

  def __del__(self):
    self._sess.close()     