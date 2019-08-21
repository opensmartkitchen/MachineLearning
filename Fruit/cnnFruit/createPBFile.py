import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os,sys
from tensorflow.python.framework import graph_util

with tf.Graph().as_default():
  with tf.Session() as sess:

   # Code as is creates and restores model and saves graph as pbtxt
    #.....
    

    #restore model
    output_node_names="y_pred"   
    saver = tf.train.import_meta_graph("model.ckpt.meta", clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, "model.ckpt")
    #.....
     # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess,'./tensorflowModel.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), '.', 'graph.pbtxt', as_text=True)
    
    