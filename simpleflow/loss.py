#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Loss Objects.
'''
from .operations import Operation
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
import torch
from torch.autograd import Variable as V

# ------------------------------------------------------------------------------
# Mean Squared Error
# ------------------------------------------------------------------------------

class MeanSquaredError(Operation):
    ''' A Mean Squared Error operation.
    '''
    def __init__(self, x, y, name=None, framework='tf'):
        ''' MeanSquaredError constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def tf_compute_output(self):
        mse = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
        return mse

    def pytorch_compute_output(self):
        mse = torch.nn.MSELoss(reduction='mean')
        return mse

    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes
        #self.output_value = np.add(x.output_value, y.output_value)
        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(
            	x.output_value, y.output_value).numpy() #* x.output_value.shape[0]
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(x.output_value), torch.from_numpy(y.output_value)).numpy()
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the addition output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            tf_y = tf.constant(y, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch((tf_x, tf_y))
                # TODO(gudiandian): fix the batch_size problem
                # out = grad * x.shape[0] * self.tf_compute_output()(tf_x, tf_y)
                out = grad * self.tf_compute_output()(tf_x, tf_y)
            dz_dy = g.gradient(out, tf_y).numpy()
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return [dz_dx, dz_dy]
        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_y = V(torch.from_numpy(np.array(y)), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_y)
            out.backward(th_grad, retain_graph=True)
            return [np.array(th_x.grad), np.array(th_y.grad)]
        else:
            raise NotImplementedError

def mse(x, y, name=None, framework='tf'):
    ''' Returns mse(x, y).
    '''
    return MeanSquaredError(x, y, name, framework=framework)


# ------------------------------------------------------------------------------
# Categorical Crossentropy Error
# ------------------------------------------------------------------------------

class CategoricalCrossentropy(Operation):
    ''' A Categorical Crossentropy operation.
    '''
    def __init__(self, x, y, name=None, framework='tf'):
        ''' CategoricalCrossentropy constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def tf_compute_output(self):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce

    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes
        #self.output_value = np.add(x.output_value, y.output_value)
        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(
            	x.output_value, y.output_value).numpy()
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the addition output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            tf_y = tf.constant(y, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch((tf_x, tf_y))
                out = grad * self.tf_compute_output()(tf_x, tf_y)
            dz_dy = g.gradient(out, tf_y).numpy()
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return [dz_dx, dz_dy]
        else:
            raise NotImplementedError

def cce(x, y, name=None, framework='tf'):
    ''' Returns cce(x, y).
    '''
    return CategoricalCrossentropy(x, y, name, framework=framework)

# ------------------------------------------------------------------------------
# Negative Log Likelihood
# ------------------------------------------------------------------------------

class NegativeLogLikelihood(Operation):
    ''' A Negative Log Likelihood operation.
    '''
    def __init__(self, x, y, name=None, framework='pytorch'):
        ''' NegativeLogLikelihood constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def pytorch_compute_output(self):
        nll = torch.nn.NLLLoss()
        return nll


    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes
        if self.framework == 'pytorch':
            print(x.output_value.shape, y.output_value.shape)
            input = torch.autograd.Variable(torch.from_numpy(x.output_value))
            target = torch.autograd.Variable(torch.from_numpy(y.output_value))
            self.output_value = self.pytorch_compute_output()(input, target).numpy()
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the addition output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        if self.framework == 'pytorch':
            th_x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=True)
            th_y = torch.autograd.Variable(torch.from_numpy(np.array(y)), requires_grad=True)
            th_grad = torch.autograd.Variable(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_y)
            out.backward(th_grad, retain_graph=True)
            return [np.array(th_x.grad), np.array(th_y.grad)]
        else:
            raise NotImplementedError

def nll(x, y, name=None, framework='tf'):
    ''' Returns nll(x, y).
    '''
    return NegativeLogLikelihood(x, y, name, framework=framework)

