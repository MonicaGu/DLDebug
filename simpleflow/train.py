#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optimizer classes for parameters optimization.
'''
from .operations import Operation, compute_gradients
import numpy as np
import tensorflow as tf
import torch
import torch.optim as optim
from torch.autograd import Variable as V

class GradientDescentOptimizer(object):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate, framework=None):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        self.learning_rate = learning_rate
        self.framework = framework

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        learning_rate = self.learning_rate
        framework = self.framework

        class MinimizationOperation(Operation):
            def compute_output(self):
                if framework == 'tf':
                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                    # Get gradient table.
                    #print("in train: ", optimizer)
                    grad_table = compute_gradients(loss, optimizer=optimizer)

                    # Iterate all trainable variables in graph.
                    for var in DEFAULT_GRAPH.trainable_variables:
                        if var in grad_table:
                            grad = grad_table[var]

                        # Update its output value.  
                        tmpvar = tf.Variable(tf.constant(var.output_value), tf.float64)
                        optimizer.apply_gradients(zip([grad], [tmpvar]))
                        var.output_value = tmpvar.numpy()
                
                elif framework == 'pytorch':
                    # Get gradient table.
                    grad_table = compute_gradients(loss)

                    # Iterate all trainable variables in graph.
                    for var in DEFAULT_GRAPH.trainable_variables:
                        if var in grad_table:
                            grad = grad_table[var]

                        # Update its output value.  
                        tmpvar = V(torch.from_numpy(np.array(var.output_value)), requires_grad=True)
                        tmpgrad = V(torch.from_numpy(np.array(grad)))
                        tmpvar.grad = tmpgrad
                        optimizer = optim.SGD([{'params': tmpvar}], lr=learning_rate)
                        optimizer.step()
                        var.output_value = tmpvar.detach().numpy()
                
                else:
                    # Get gradient table.
                    grad_table = compute_gradients(loss)

                    # Iterate all trainable variables in graph.
                    for var in DEFAULT_GRAPH.trainable_variables:
                        if var in grad_table:
                            grad = grad_table[var]

                        # Update its output value.
                        var.output_value -= learning_rate*grad

        return MinimizationOperation()

