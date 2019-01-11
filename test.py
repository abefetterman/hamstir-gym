import tensorflow as tf
import numpy as np
from PIL import Image

import argparse
import numpy as np

tf.reset_default_graph()
with tf.gfile.GFile('./models/graph.pb', 'rb') as f:
   graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:
    result = tf.import_graph_def(graph_def, return_elements=['model/pi/add:0'])
    # print('Operations in Optimized Graph:')
    # print([op.name for op in G.get_operations()]) 
    input = G.get_tensor_by_name('import/model/module_apply_default/hub_input/Sub:0')
    output = G.get_tensor_by_name('import/model/pi/add:0')
    
    image = Image.open('./models/raspi1.jpg')
    image = np.reshape(np.array(image), [1,192,192,3])
    image = (image/255.0)*2 - 1.0
    print(sess.run(output, feed_dict={input:image}))
    image = Image.open('./models/raspi2.jpg')
    image = np.reshape(np.array(image), [1,192,192,3])
    image = (image/255.0)*2 - 1.0
    print(sess.run(output, feed_dict={input:image}))
    image = Image.open('./models/raspi3.jpg')
    image = np.reshape(np.array(image), [1,192,192,3])
    image = (image/255.0)*2 - 1.0
    print(sess.run(output, feed_dict={input:image}))
    image = Image.open('./models/raspi4.jpg')
    image = np.reshape(np.array(image), [1,192,192,3])
    image = (image/255.0)*2 - 1.0
    print(sess.run(output, feed_dict={input:image}))