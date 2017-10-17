import itertools

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plot import *

def main():

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1)[:, np.newaxis]

    def cross_entropy(y,y_pred):
        return - np.sum(np.multiply(y, np.log(y_pred)))

    def test_accuracy(y,y_pred):
        correct = np.equal(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)).sum()
        return correct/100

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 100, 784, 100, 10

    # Randomly initialize weights
    w1_0 = np.random.randn(D_in, H)*0.1
    w2_0 = np.random.randn(H, D_out)*0.1
    B = np.random.randn(H,D_out)#w2_0.copy()*100
    B_big = B*100 # excel 100 times, equivalent to say w1's leanring rate *100
    B_small = B*0.01 # shrink 100 times, equivalent to say w1's leanring rate *0.01
    
    test = [B,B_big,B_small]

    learning_rate = 1e-4
    loss_history = []
    accuracy_history = []
    for i in range(len(test)+1):
        mnist = input_data.read_data_sets('mnist',
                                    one_hot=True)
        w1 = w1_0.copy() #deep copy
        assert(np.array_equal(w1,w1_0)) # assert reset
        w2 = w2_0.copy() #deep copy
        assert(np.array_equal(w2,w2_0)) # assert reset

        def get_batch(train,batch_size=100):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if train:
              xs, ys = mnist.train.next_batch(batch_size)
            else:
              rows = np.random.randint(1000,size=batch_size) 
              xs, ys = mnist.test.images[rows,:], mnist.test.labels[rows,:]
              #xs, ys = mnist.test.images[:batch_size,:], mnist.test.labels[:batch_size,:]
            return xs, ys

        _loss = []
        _accuracy = []
        for t in range(6000):

            if i == 0 and t > 2000:
                break
            
            x,y = get_batch(True,100)
            # Forward pass: compute predicted y
            h = x.dot(w1)
            h_relu = np.maximum(h, 0)
            y_pred = softmax(h_relu.dot(w2))
            # Compute and print loss
            loss = cross_entropy(y,y_pred) / 100

            x_test,y_test = get_batch(False)
            h_test = x_test.dot(w1)
            h_relu_test = np.maximum(h_test, 0)
            y_pred_test = softmax(h_relu_test.dot(w2))
            accuracy = test_accuracy(y_test,y_pred_test)
            _loss.append(loss)
            _accuracy.append(accuracy)
            print(t,loss,accuracy)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = (y_pred - y)
            grad_w2 = h_relu.T.dot(grad_y_pred)
            if i == 0:
                grad_h_relu = grad_y_pred.dot(w2.T)
            else:
                grad_h_relu = grad_y_pred.dot(test[i-1].T)
                trace = np.trace(grad_y_pred.dot(w2.T).dot(test[i-1]).dot(grad_y_pred.T))/100
                if trace < 0: print('Not aligned')
            grad_h = grad_h_relu.copy()
            grad_h[h < 0] = 0
            grad_w1 = x.T.dot(grad_h)

            # Update weights
            '''
            if i != 0:
                if t<2000:
                    w1 -= learning_rate * grad_w1
                elif t< 4000:
                    w2 -= learning_rate * grad_w2
                else:
                    w1 -= learning_rate * grad_w1                   
            else:
                w1 -= learning_rate * grad_w1
                w2 -= learning_rate * grad_w2'''
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
                
        loss_history.append(_loss)
        accuracy_history.append(_accuracy)
        
    plot_learning_curve('Learning Curve',accuracy_history,loss_history,['BP','FA','FA-big','FA-small'],ylim=None)

if __name__ == '__main__':
    main()
