"""Try to predict 7 random examples from test data"""

import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessing
from model import Model
from train import predictions

x = preprocessing.get_x("data/test")
indices = np.random.choice(len(x), 7)
examples = x[indices]
model = Model()

with tf.Session(graph=model.graph) as session:

    # Restore model with 82% accuracy on test data
    model.saver.restore(session, "./try2.ckpt")
    
    log_1, log_2, log_3, log_4, log_5 = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                    model.logits_4, model.logits_5],
                                                    feed_dict={model.x: examples, model.keep_prob: 1.0})
    
# Make predictions on 7 random examples from test data
predictions = predictions(log_1, log_2, log_3, log_4, log_5)

# Create matplotlib plot for visualization 
plt.rcParams['figure.figsize'] = (20.0, 20.0)
f, ax = plt.subplots(nrows=1, ncols=7)
for i, el in enumerate(indices):
    image = scipy.misc.imread("data/test/"+str(el+1)+".png", flatten=False)
    ax[i].axis('off')
    number = predictions[i][predictions[i] < 10]
    ax[i].set_title("Pred: "+''.join(number.astype("str")), loc='center')
    ax[i].imshow(image)    
plt.show()
