import numpy as np
import tensorflow as tf

import preprocessing
from model import Model

def load_data():
    """Loads data"""
    x_train = preprocessing.get_x("data/train")
    y_train = preprocessing.get_y1("data/train")
    x_test = preprocessing.get_x("data/test")
    y_test = preprocessing.get_y1("data/test")
    return x_train, y_train, x_test, y_test


def predictions(logit_1,logit_2,logit_3,logit_4,logit_5):
    """Converts predictions into understandable format.
    For example correct prediction for 2 will be > [2,10,10,10,10]
    """
    first_digits = np.argmax(logit_1,axis=1)
    second_digits =  np.argmax(logit_2,axis=1)
    third_digits =  np.argmax(logit_3,axis=1)
    fourth_digits = np.argmax(logit_4,axis=1)
    fifth_digits = np.argmax(logit_5,axis=1)
    stacked_digits = np.vstack((first_digits,second_digits,third_digits,fourth_digits,fifth_digits))
    rotated_digits = np.rot90(stacked_digits)[::-1]
    return rotated_digits


def accuracy(logit_1,logit_2,logit_3,logit_4,logit_5,y_):
    """Computes accuracy"""
    correct_prediction = []
    y_ = y_[:,1:]
    rotated_digits = predictions(logit_1, logit_2, logit_3, logit_4, logit_5)
    for e in range(len(y_)):
        if np.array_equal(rotated_digits[e],y_[e]):
            correct_prediction.append(True)
        else:
            correct_prediction.append(False)       
    return (np.mean(correct_prediction))*100.0        


def train(batch_size=64, number_of_iterations=100000, reuse=False):
    """Trains CNN."""
    x_train, y_train, x_test, y_test = load_data()
    print "Data uploaded!"
    
    model = Model()
    with tf.Session(graph=model.graph) as session:
        z, n = 0, 0
        tf.global_variables_initializer().run()

        # Change to True, if you want to restore the model with 82% test accuracy
        if reuse:
            model.saver.restore(session, "./try2.ckpt")
                
        for i in range(number_of_iterations):
            indices= np.random.choice(len(y_train), batch_size)
            bat_x = x_train[indices]
            bat_y = y_train[indices]
            _, l= session.run([model.optimizer,model.loss], feed_dict={model.x: bat_x, model.y: bat_y,
                                                                       model.keep_prob: 0.5})
            # Check batch accuracy and loss
            if (i % 500 == 0):
                log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                            model.logits_4, model.logits_5, model.y],
                                                                    feed_dict={model.x: bat_x, model.y: bat_y,
                                                                               model.keep_prob: 1.0})
                print "Iteration number: {}".format(i)
                print "Batch accuracy: {},  Loss: {}".format(accuracy(log_1, log_2, log_3, log_4, log_5, y_), l)

        # Evaluate accuracy by parts, if you use GPU and it has low memory.
        # For example, I have 2 GB GPU memory and I need to feed test data by parts(six times by 2178 examples)
        for el in range(6):
            log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                        model.logits_4, model.logits_5, model.y],
                                                            feed_dict={model.x: x_test[n:n+2178], model.y: y_test[n:n+2178],
                                                                       model.keep_prob: 1.0})
            n += 2178
            
            # Combine accuracy
            z += accuracy(log_1, log_2, log_3, log_4, log_5, y_)
            
        print "Test accuracy: {}".format((z/6.0))    

        # Save model in file "try1.ckpt"
        model.saver.save(session, "./try1.ckpt")    

if __name__ == '__main__':
    train()

