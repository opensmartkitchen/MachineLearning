import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from google.protobuf import text_format

#with open('/tmp/myfile.pbtxt') as f:
with open('graph.pbtxt') as f:
  txt = f.read()
gdef = text_format.Parse(txt, tf.GraphDef())

#tf.train.write_graph(gdef, '/tmp', 'myfile.pb', as_text=False)
tf.train.write_graph(gdef, '.','myfile.pb', as_text=False)