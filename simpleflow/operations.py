#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Operation classes in computational graph.
'''
from queue import Queue
from memory_profiler import profile
import numpy as np

import tensorflow as tf
from torch.autograd import Variable as V
import torch
import mxnet as mx

class Operation(object):
    ''' Base class for all operations in DLdebug.

    An operation is a node in computational graph receiving zero or more nodes
    as input and produce zero or more nodes as output. Vertices could be an
    operation, variable or placeholder.
    '''
    def __init__(self, *input_nodes, name=None, framework='tf'):
        ''' Operation constructor.

        :param input_nodes: Input nodes for the operation node.
        :type input_nodes: Objects of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        # Nodes received by this operation.
        self.input_nodes = input_nodes

        # Nodes that receive this operation node as input.
        self.output_nodes = []

        # Output value of this operation in session execution.
        self.output_value = None

        # Operation name.
        self.name = name

        # Graph the operation belongs to.
        self.graph = DEFAULT_GRAPH

        # The framework used in inference and gradient calculation. 
        # Use TensorFlow by default.
        self.framework = framework

        # Add this operation node to destination lists in its input nodes.
        for node in input_nodes:
            node.output_nodes.append(self)

        # Add this operation to default graph.
        self.graph.operations.append(self)

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient of the operation wrt inputs.
        '''
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

# ------------------------------------------------------------------------------
# Addition operation
# ------------------------------------------------------------------------------

class Add(Operation):
    ''' An addition operation.
    '''
    def __init__(self, x, y, name=None, framework='tf'):
        ''' Addition constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def tf_compute_output(self):
        return tf.add

    def pytorch_compute_output(self):
        return torch.add

    def mx_compute_output(self, x, y):
        x0 = mx.nd.array(x).as_np_ndarray()
        y0 = mx.nd.array(y).as_np_ndarray()
        return (x0 + y0).asnumpy()

    def tflite_compute_output(self, x, y):
        input1 = tf.keras.layers.Input(shape=np.array(x).shape, dtype=tf.float32)
        input2 = tf.keras.layers.Input(shape=np.array(y).shape, dtype=tf.float32)
        output = tf.add(input1, input2)
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index1 = interp.get_input_details()[0]['index']
        input_index2 = interp.get_input_details()[1]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index1, np.array(x).shape)
        interp.resize_tensor_input(input_index2, np.array(y).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index1, np.array(x).astype(np.float32))
        interp.set_tensor(input_index2, np.array(y).astype(np.float32))
        interp.invoke()
        return interp.get_tensor(output_index)

    @profile(precision=4)
    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, y.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value)),
                torch.from_numpy(np.array(y.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value, y.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value, y.output_value)
            return self.output_value
        else:
            raise NotImplementedError
    @profile(precision=4)
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

            while np.ndim(dz_dx) > len(np.shape(x)):
                dz_dx = np.mean(dz_dx, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    dz_dx = np.mean(dz_dx, axis=axis, keepdims=True)       
        
            while np.ndim(dz_dy) > len(np.shape(y)):
                dz_dy = np.mean(dz_dy, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    dz_dy = np.mean(dz_dy, axis=axis, keepdims=True)
            return [dz_dx, dz_dy]
        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_y = V(torch.from_numpy(np.array(y)), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_y)
            out.backward(th_grad, retain_graph=True)
            return [np.array(th_x.grad), np.array(th_y.grad)]
        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            y0 = mx.nd.array(y).as_np_ndarray()
            x0.attach_grad()
            y0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * (x0 + y0)
                mx_result.backward()
            return [x0.grad.asnumpy(), y0.grad.asnumpy()]
        else:
            raise NotImplementedError


def add(x, y, name=None, framework='tf'):
    ''' Returns x + y element-wise.
    '''
    return Add(x, y, name, framework=framework)

# ------------------------------------------------------------------------------
# Multiplication operation
# ------------------------------------------------------------------------------

class Multiply(Operation):
    ''' Multiplication operation.
    '''
    def __init__(self, x, y, name=None, framework='tf'):
        ''' Multiplication constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def tf_compute_output(self):
        return tf.multiply

    def pytorch_compute_output(self):
        return torch.mul

    def mx_compute_output(self, x, y):
        x0 = mx.nd.array(x).as_np_ndarray()
        y0 = mx.nd.array(y).as_np_ndarray()
        return (x0 * y0).asnumpy()

    def tflite_compute_output(self, x, y):
        input1 = tf.keras.layers.Input(shape=np.array(x).shape, dtype=tf.float32)
        input2 = tf.keras.layers.Input(shape=np.array(y).shape, dtype=tf.float32)
        output = tf.multiply(input1, input2)
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index1 = interp.get_input_details()[0]['index']
        input_index2 = interp.get_input_details()[1]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index1, np.array(x).shape)
        interp.resize_tensor_input(input_index2, np.array(y).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index1, np.array(x).astype(np.float32))
        interp.set_tensor(input_index2, np.array(y).astype(np.float32))
        interp.invoke()
        return interp.get_tensor(output_index)

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, y.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value)),
                torch.from_numpy(np.array(y.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value, y.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value, y.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the mutiply output.
        :type grad: number or a ndarray.
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

            while np.ndim(dz_dx) > len(np.shape(x)):
                dz_dx = np.mean(dz_dx, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    dz_dx = np.mean(dz_dx, axis=axis, keepdims=True)

            while np.ndim(dz_dy) > len(np.shape(y)):
                dz_dy = np.mean(dz_dy, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    dz_dy = np.mean(dz_dy, axis=axis, keepdims=True)

            return [dz_dx, dz_dy]
        
        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_y = V(torch.from_numpy(np.array(y)), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_y) # th_x * th_y
            out.backward(th_grad, retain_graph=True)
            return [np.array(th_x.grad), np.array(th_y.grad)]

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            y0 = mx.nd.array(y).as_np_ndarray()
            x0.attach_grad()
            y0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * x0 * y0
                mx_result.backward()
            return [x0.grad.asnumpy(), y0.grad.asnumpy()]

        else:
            raise NotImplementedError


def multiply(x, y, name=None, framework='tf'):
    ''' Returns x * y element-wise.
    '''
    return Multiply(x, y, name, framework=framework)

# ------------------------------------------------------------------------------
# Matrix multiplication operation
# ------------------------------------------------------------------------------

class MatMul(Operation):
    ''' Matrix multiplication operation.
    '''
    def __init__(self, x, y, name=None, framework='tf'):
        ''' MatMul constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name, framework=framework)

    def tf_compute_output(self):
        return tf.matmul

    def pytorch_compute_output(self):
        return torch.mm

    def mx_compute_output(self, x, y):
        x0 = mx.nd.array(x).as_np_ndarray()
        y0 = mx.nd.array(y).as_np_ndarray()
        return (mxnet.ndarray.batch_dot(x0, y0)).asnumpy()

    def tflite_compute_output(self, x, y):
        input1 = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        input2 = tf.keras.layers.Input(shape=y.shape, dtype=tf.float32)
        output = tf.matmul(input1, input2)
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index1 = interp.get_input_details()[0]['index']
        input_index2 = interp.get_input_details()[1]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index1, np.array([x]).shape)
        interp.resize_tensor_input(input_index2, np.array([y]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index1, np.array([x]).astype(np.float32))
        interp.set_tensor(input_index2, np.array([y]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, y.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value)),
                torch.from_numpy(np.array(y.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value, y.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value, y.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for matrix multiplication.

        :param grad: The gradient of other operation wrt the matmul output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x, y = [node.output_value for node in self.input_nodes]

        # Default gradient wrt the matmul output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
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

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_y = V(torch.from_numpy(np.array(y)), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_y) # th_x * th_y
            out.backward(th_grad, retain_graph=True)
            return [np.array(th_x.grad), np.array(th_y.grad)]

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            y0 = mx.nd.array(y).as_np_ndarray()
            x0.attach_grad()
            y0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.batch_dot(x0, y0)
                mx_result.backward()
            return [x0.grad.asnumpy(), y0.grad.asnumpy()]

        else:
            raise NotImplementedError


def matmul(x, y, name=None, framework='tf'):
    ''' Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
    '''
    return MatMul(x, y, name, framework=framework)

# ------------------------------------------------------------------------------
# Sigmoid operation
# ------------------------------------------------------------------------------

class Sigmoid(Operation):
    ''' Sigmoid operation.
    '''
    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def tf_compute_output(self):
        return tf.sigmoid

    def pytorch_compute_output(self):
        return torch.nn.Sigmoid

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.sigmoid(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.sigmoid(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the sigmoid operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for sigmoid.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the sigmoid output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.sigmoid(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError


def sigmoid(x, name=None):
    ''' Computes sigmoid of `x` element-wise.
    '''
    return Sigmoid(x, name=name)

# ------------------------------------------------------------------------------
# Logarithm operation
# ------------------------------------------------------------------------------

class Log(Operation):
    ''' Natural logarithm operation.
    '''
    def __init__(self, x, name=None):
        ''' Logarithm constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def tf_compute_output(self):
        return tf.math.log

    def pytorch_compute_output(self):
        return torch.log

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.log(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.math.log(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the log operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for log.

        :param grad: The gradient of other operation wrt the log output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the log output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.log(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError

def log(x, name=None):
    ''' Computes the natural logarithm of x element-wise.
    '''
    return Log(x, name=name)

# ------------------------------------------------------------------------------
# Negative operation
# ------------------------------------------------------------------------------

class Negative(Operation):
    ''' Negative operation.
    '''
    def __init__(self, x, name=None):
        ''' Negative constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of negative function.
        '''
        x, = self.input_nodes
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for negative operation wrt input value.

        :param grad: The gradient of other operation wrt the negative output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -grad

# ------------------------------------------------------------------------------
# Reduce sum operation
# ------------------------------------------------------------------------------

class ReduceSum(Operation):
    ''' Reduce sum operation.
    '''
    def __init__(self, x, axis=None, keepdims=False):
        ''' ReduceSum constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param axis: The dimensions to reduce. If `None`, reduces all dimensions.
        :type axis: int.
        '''
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def tf_compute_output(self):
        return tf.reduce_sum

    def tflite_compute_output(self, x, rate, keepdims):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.reduce_sum(input1, input2)
        model = tf.keras.Model(inputs=input1, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the value of reduce_sum function.
        '''
        x, = self.input_nodes
        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, 
                self.axis, self.keepdims).numpy()
            return self.output_value

        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value,
                self.axis, self.keepdims)
            return self.output_value

        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for reduce_sum operation wrt input value.

        :param grad: The gradient of other operation wrt the reduce_sum output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(input_value, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(input_value)
                out = grad * self.tf_compute_output()(tf_x, self.axis, self.keepdims)
            dz_dx = g.gradient(out, tf_x).numpy()

            del g
            return dz_dx
        else:
            raise NotImplementedError

def reduce_sum(x, axis=None):
    ''' Computes the sum of elements across dimensions of a tensor.
    '''
    return ReduceSum(x, axis=axis)

# ------------------------------------------------------------------------------
# Square operation
# ------------------------------------------------------------------------------

class Square(Operation):
    ''' Square operation.
    '''
    def __init__(self, x, name=None):
        ''' Square constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def tf_compute_output(self):
        return tf.math.square

    def pytorch_compute_output(self):
        # no square function in pytorch. Use torch.mul(x, x) instead.
        return torch.mul

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.square(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.math.square(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the square operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value)), 
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for square.

        :param grad: The gradient of other operation wrt the square output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the square output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.square(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError

def square(x, name=None):
    ''' Computes square of x element-wise.
    '''
    return Square(x, name=name)

# ------------------------------------------------------------------------------
# Reduce mean operation
# ------------------------------------------------------------------------------

class ReduceMean(Operation):
    ''' Reduce sum operation.
    '''
    def __init__(self, x, axis=None, keepdims=False):
        ''' ReduceMean constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param axis: The dimensions to reduce. If `None`, reduces all dimensions.
        :type axis: int.
        '''
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def tf_compute_output(self):
        return tf.reduce_mean

    def tflite_compute_output(self, x, rate, keepdims):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.reduce_mean(input1, input2)
        model = tf.keras.Model(inputs=input1, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the value of reduce mean function.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, 
                self.axis, self.keepdims).numpy()
            return self.output_value

        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value,
                self.axis, self.keepdims)
            return self.output_value

        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for reduce mean operation wrt input value.

        :param grad: The gradient of other operation wrt the reduce mean output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(input_value, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(input_value)
                out = grad * self.tf_compute_output()(tf_x, self.axis, self.keepdims)
            dz_dx = g.gradient(out, tf_x).numpy()

            del g
            return dz_dx
        else:
            raise NotImplementedError


def reduce_mean(x, axis=None):
    ''' Computes the mean of elements across dimensions of a tensor.
    '''
    return ReduceMean(x, axis=axis)

# ------------------------------------------------------------------------------
# Softmax operations
# ------------------------------------------------------------------------------

class Softmax(Operation):
    ''' Softmax operation.
    '''
    def __init__(self, x, name=None, framework='tf'):
        ''' Softmax constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name, framework=framework)

    def tf_compute_output(self):
        return tf.nn.softmax
    
    def pytorch_compute_output(self):
        return torch.nn.functional.softmax

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.Softmax(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=np.array(x).shape, dtype=tf.float32)
        output = tf.nn.softmax(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, x.shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, x.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(output_index)

    def compute_output(self):
        ''' Compute and return the value of softmax function.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(x.output_value), dim=1).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for softmax operation wrt input value.

        :param grad: The gradient of other operation wrt the softmax output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)
        
        if self.framework == 'tf':
            tf_x = tf.constant(input_value, tf.float64)
            with tf.GradientTape() as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(input_value), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, dim=1)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.Softmax(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError

        

def softmax(x, name=None, framework='tf'):
    ''' Computes softmax of x.
    '''
    return Softmax(x, name=name, framework=framework)

def div(x, y, name=None):
    return multiply(x, inv(y))


# ------------------------------------------------------------------------------
# Inv operation
# ------------------------------------------------------------------------------

class Inv(Operation):
    ''' Inv operation.
    '''
    def __init__(self, x, name=None):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def tf_compute_output(self):
        return tf.math.reciprocal

    def pytorch_compute_output(self):
        return torch.reciprocal

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.reciprocal(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.math.reciprocal(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for reciprocal.

        :param grad: The gradient of other operation wrt the inv output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the inv output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.reciprocal(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError


def inv(x, name=None):
    ''' Computes the inv of a tensor.
    '''
    return Inv(x, name=name)

# ------------------------------------------------------------------------------
# Exp operation
# ------------------------------------------------------------------------------

class Exp(Operation):
    ''' Exp operation.
    '''
    def __init__(self, x, name=None):
        ''' Exp constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def tf_compute_output(self):
        return tf.math.exp

    def pytorch_compute_output(self):
        return torch.exp

    def mx_compute_output(self, x):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.exp(x0)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.math.exp(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the exp operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for exp.

        :param grad: The gradient of other operation wrt the exp output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the exp output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.exp(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError

def exp(x, name=None):
    ''' Computes the inv of a tensor.
    '''
    return Exp(x, name=name)

# ------------------------------------------------------------------------------
# Transpose operation
# ------------------------------------------------------------------------------

class Transpose(Operation):
    ''' Transpose operation. PERMS NOT IMPLEMENTED!
    '''
    def __init__(self, x, axes=None, name=None):
        ''' Transpose constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)
        self.axes = axes

    def tf_compute_output(self):
        return tf.transpose

    def mx_compute_output(self, x, axes):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.transpose(x0, axes=axes)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.math.tranposes(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the transpose operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value, perm=self.axes).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = (torch.from_numpy(
                np.array(x.output_value)).permute(self.axes)).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value, axes=axes)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for transpose.

        :param grad: The gradient of other operation wrt the transpose output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the transpose output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x, perm=self.axes)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = th_x.permute(self.axes)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.transpose(x0, axes=self.axes)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError


def transpose(x, name=None):
    ''' Computes the inv of a tensor.
    '''
    return Transpose(x, name=name)

# ------------------------------------------------------------------------------
# Softmax operation
# ------------------------------------------------------------------------------

class Dropout(Operation):
    ''' Square operation.
    '''
    def __init__(self, x, rate, name=None, framework='tf'):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        self.rate = rate
        super(self.__class__, self).__init__(x, name=name, framework=framework)

    def tf_compute_output(self):
        return tf.nn.dropout
    
    def pytorch_compute_output(self):
        return torch.nn.Dropout

    def mx_compute_output(self, x, rate):
        x0 = mx.nd.array(x).as_np_ndarray()
        return mx.gluon.nn.Dropout(rate=rate)(x0).asnumpy()

    def tflite_compute_output(self, x, rate):
        input = tf.keras.layers.Input(shape=np.array(x).shape, dtype=tf.float32)
        if tf.__version__ > '2':
            output = tf.nn.dropout(input, rate)
        else:
            output = tf.nn.dropout(input, 1 - rate)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, x.shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, x.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(output_index)

    def compute_output(self):
        ''' Compute and return the value of dropout function.
        '''
        x, = self.input_nodes
        if self.framework == 'tf':
            # API change
            if tf.__version__ > '2':
                self.output_value = self.tf_compute_output()(
                    x.output_value, self.rate).numpy()
            else:
                self.output_value = self.tf_compute_output()(
                    x.output_value, 1 - self.rate).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(self.rate)(
                torch.from_numpy(x.output_value)).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value, self.rate)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value, self.rate)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for dropout operation wrt input value.

        :param grad: The gradient of other operation wrt the dropout output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value) 
        
        if self.framework == 'tf':
            if tf.__version__ > '2':
                tf_x = tf.constant(input_value, tf.float64)
                with tf.GradientTape() as g:
                    g.watch(tf_x)
                    out = grad * self.tf_compute_output()(tf_x, self.rate)
                dz_dx = g.gradient(out, tf_x).numpy()
            else:
                tf_x = tf.constant(input_value, tf.float64)
                with tf.GradientTape() as g:
                    g.watch(tf_x)
                    out = grad * self.tf_compute_output()(tf_x, 1 - self.rate)
                dz_dx = g.gradient(out, tf_x).numpy()
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(input_value), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x, self.rate)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)
        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = mx.gluon.nn.Dropout(rate=self.rate)(x0)
                mx_result.backward()
            return x0.grad.asnumpy()
        else:
            raise NotImplementedError

def dropout(x, rate, name=None, framework='tf'):
    ''' Computes the inv of a tensor.
    '''
    return Dropout(x, rate=rate, name=name, framework=framework)


# ------------------------------------------------------------------------------
# ReLU operation
# ------------------------------------------------------------------------------

class ReLU(Operation):
    ''' ReLU operation.
    '''
    def __init__(self, x, axes=None, name=None):
        ''' ReLU constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)
        self.axes = axes

    def pytorch_compute_output(self):
        return torch.nn.ReLU

    def tf_compute_output(self):
        return tf.nn.relu

    def mx_compute_output(self, x, axes):
        x0 = mx.nd.array(x).as_np_ndarray()
        return (mxnet.ndarray.relu(x0, axes=axes)).asnumpy()

    def tflite_compute_output(self, x):
        input = tf.keras.layers.Input(shape=x.shape, dtype=tf.float32)
        output = tf.nn.relu(input)
        model = tf.keras.Model(inputs=input, outputs=output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        interp = tf.lite.Interpreter(model_content=tflite_model)

        input_index = interp.get_input_details()[0]['index']
        output_index = interp.get_output_details()[0]['index']

        interp.resize_tensor_input(input_index, np.array([x]).shape)
        interp.allocate_tensors()
        interp.set_tensor(input_index, np.array([x]).astype(np.float32))
        interp.invoke()

        return interp.get_tensor(output_index)[0]

    def compute_output(self):
        ''' Compute and return the relu operation result.
        '''
        x, = self.input_nodes

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(x.output_value).numpy()
            return self.output_value
        elif self.framework == 'pytorch':
            self.output_value = self.pytorch_compute_output()(
                torch.from_numpy(np.array(x.output_value))).numpy()
            return self.output_value
        elif self.framework == 'tflite':
            self.output_value = self.tflite_compute_output(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(x.output_value)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient for relu.

        :param grad: The gradient of other operation wrt the relu output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x = self.input_nodes[0].output_value

        # Default gradient wrt the relu output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_x = tf.constant(x, tf.float64)
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_x)
                out = grad * self.tf_compute_output()(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            del g
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(x), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.pytorch_compute_output()(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * mxnet.ndarray.relu(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError


def relu(x, name=None):
    ''' Computes the ReLU of a tensor.
    '''
    return ReLU(x, name=name)

# ------------------------------------------------------------------------------
# EinSum operation
# ------------------------------------------------------------------------------

class EinSum(Operation):
    ''' EinSum operation.
    '''
    def __init__(self, x, equation=None, name=None):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)
        self.equation = equation

    def tf_compute_output(self):
        return tf.einsum

    def mx_compute_output(self, inputs, equation):
        xs = mx.nd.array(x).as_np_ndarray()
        xs = [mx.nd.array(x).as_np_ndarray() for x in inputs]
        return (mx.np.eimsum(equation, xs)).asnumpy()

    def compute_output(self):
        ''' Compute and return the einsum operation result.
        '''
        inputs = [node.output_value for node in self.input_nodes]

        if self.framework == 'tf':
            self.output_value = self.tf_compute_output()(self.equation, inputs).numpy()
            return self.output_value
        elif self.framework == 'mx':
            self.output_value = self.mx_compute_output(self.equation, inputs)
            return self.output_value
        else:
            raise NotImplementedError

    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute and return the gradient einsum.

        :param grad: The gradient of other operation wrt the eimsum output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        inputs = [node.output_value for node in self.input_nodes]

        # Default gradient wrt the sinsum output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        if self.framework == 'tf':
            tf_xs = [tf.constant(x, tf.float64) for x in inputs]
            with tf.GradientTape(persistent=True) as g:
                g.watch(tf_xs)
                out = grad * self.tf_compute_output()(self.equation, tf_xs)
            dz_dxs = [g.gradient(out, tf_x).numpy() for tf_x in tf_xs]
            del g
            return dz_dxs

        elif self.framework == 'mx':
            xs = [mx.nd.array(x).as_np_ndarray().attach_grad() for x in inputs]
            with mx.autograd.record():
                mx_result = grad * mx.np.eimsum(equation, xs)
                mx_result.backward()
            return [x.grad.asnumpy() for x in xs]

        else:
            raise NotImplementedError


def einsum(x, name=None):
    ''' Computes the einsum of a tensor.
    '''
    return EinSum(x, name=name)


# ------------------------------------------------------------------------------
# Custom operation
# ------------------------------------------------------------------------------

class Custom(Operation):
    ''' Transpose operation.
    '''
    def __init__(self, model, x, name=None, framework='tf'):
        ''' Operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)
        self.model = model

    def compute_output(self):
        ''' Compute and return the value of a custom function.
        '''
        x, = self.input_nodes
        if self.framework == 'tf' or self.framework == 'pytorch':
            self.output_value = self.model(x.output_value)
            return self.output_value
        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            return (self.model(x0)).asnumpy()
        elif self.framework == 'tflite':
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()

            interp = tf.lite.Interpreter(model_content=tflite_model)

            input_index = interp.get_input_details()[0]['index']
            output_index = interp.get_output_details()[0]['index']

            interp.resize_tensor_input(input_index, np.array([x]).shape)
            interp.allocate_tensors()
            interp.set_tensor(input_index, np.array([x]).astype(np.float32))
            interp.invoke()

            self.output_value = interp.get_tensor(output_index)[0]
            return self.output_value
        else:
            raise NotImplementedError

        
    def compute_gradient(self, grad=None, optimizer=None):
        ''' Compute the gradient for custom operation wrt input value.

        :param grad: The gradient of other operation wrt the custom op output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)
        
        if self.framework == 'tf':
            tf_x = tf.constant(input_value, tf.float64)
            #print(self.model.trainable_variables)
            with tf.GradientTape() as g:
                g.watch(tf_x)
                out = grad * self.model(tf_x)
            dz_dx = g.gradient(out, tf_x).numpy()
            with tf.GradientTape() as g:
                g.watch(self.model.trainable_variables)
                out = grad * self.model(tf_x)
            grads = g.gradient(out, self.model.trainable_variables)

            if optimizer != None:
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return dz_dx

        elif self.framework == 'pytorch':
            th_x = V(torch.from_numpy(input_value), requires_grad=True)
            th_grad = V(torch.from_numpy(grad))
            out = self.model(th_x)
            out.backward(th_grad, retain_graph=True)
            return np.array(th_x.grad)

        elif self.framework == 'mx':
            x0 = mx.nd.array(x).as_np_ndarray()
            x0.attach_grad()
            with mx.autograd.record():
                mx_result = grad * self.model(x0)
                mx_result.backward()
            return x0.grad.asnumpy()

        else:
            raise NotImplementedError


def custom(model, x, name=None):
    ''' Custom.
    '''
    return Custom(model, x, name=name)
# ------------------------------------------------------------------------------
# Constant node
# ------------------------------------------------------------------------------

class Constant(object):
    ''' Constant node in computational graph.
    '''
    def __init__(self, value, name=None):
        ''' Cosntant constructor.
        '''
        # Constant value.
        self.value = value

        # Output value of this operation in session.
        self.output_value = None

        # Nodes that receive this variable node as input.
        self.output_nodes = []

        # Operation name.
        self.name = name

        # Add to graph.
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        ''' Compute and return the constant value.
        '''
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def constant(value, name=None):
    ''' Create a constant node.
    '''
    return Constant(value, name=name)

# ------------------------------------------------------------------------------
# Variable node
# ------------------------------------------------------------------------------

class Variable(object):
    ''' Variable node in computational graph.
    '''
    def __init__(self, initial_value=None, name=None, trainable=True): 
        ''' Variable constructor.

        :param initial_value: The initial value of the variable.
        :type initial_value: number or a ndarray.

        :param name: Name of the variable.
        :type name: str.
        '''
        # Variable initial value.
        self.initial_value = initial_value

        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this variable node as input.
        self.output_nodes = []

        # Variable name.
        self.name = name

        # Graph the variable belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        ''' Compute and return the variable value.
        '''
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

# ------------------------------------------------------------------------------
# Placeholder node
# ------------------------------------------------------------------------------

class Placeholder(object):
    ''' Placeholder node in computational graph. It has to be provided a value when
        when computing the output of a graph.
    '''
    def __init__(self, name=None):
        ''' Placeholdef constructor.
        '''
        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Placeholder node name.
        self.name = name

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def placeholder(name=None):
    ''' Inserts a placeholder for a node that will be always fed.
    '''
    return Placeholder(name=name)

# ------------------------------------------------------------------------------
# Function for gradients computation.
# ------------------------------------------------------------------------------
@profile(precision=4)
def compute_gradients(target_op, optimizer=None):
    ''' Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.

    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.

    :return grad_table: A table containing node objects and gradients.
    :type grad_table: dict.
    '''
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output.
    # NOTE: It is the gradient wrt the node's OUTPUT NOT input.
    grad_table = {}

    # The gradient wrt target_op itself is 1.
    grad_table[target_op] = np.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for node traverasl.
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()

        # Compute gradient wrt the node's output.
        if node != target_op:
            grads_wrt_node_output = []

            for output_node in node.output_nodes:
                # Retrieve the gradient wrt output_node's OUTPUT.
                grad_wrt_output_node_output = grad_table[output_node]

                # Compute the gradient wrt current node's output.
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output, optimizer=optimizer)
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            # Sum all gradients wrt node's output.
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        # Put adjecent nodes to queue.
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table

