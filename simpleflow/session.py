#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Session to execute a computational graph.
'''
from functools import reduce
from memory_profiler import profile

from .operations import Operation, Variable, Placeholder, compute_gradients

import numpy as np

class Session(object):
    ''' A session to compute a particular graph.
    '''
    def __init__(self):
        ''' Session constructor.
        '''
        # Graph the session computes for.
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        ''' Context management protocal method called before `with-block`.
        '''
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Context management protocal method called after `with-block`.
        '''
        self.close()

    def close(self):
        ''' Free all output values in nodes.
        '''
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    def run(self, operation, feed_dict=None):
        ''' Compute the output of an operation.

        :param operation: A specific operation to be computed.
        :type operation: object of `Operation`, `Variable` or `Placeholder`.

        :param feed_dict: A mapping between placeholder and its actual value for the session.
        :type feed_dict: dict.
        '''
        # Get all prerequisite nodes using postorder traversal.
        postorder_nodes = _get_prerequisite(operation)
        
        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value = feed_dict[node]
            else:  # Operation and variable
                node.compute_output()

        return operation.output_value

def _get_prerequisite(operation):
    ''' Perform a post-order traversal to get a list of nodes to be computed in order.
    '''
    postorder_nodes = []

    # Collection nodes recursively.
    def postorder_traverse(operation):
        if isinstance(operation, Operation):
            for input_node in operation.input_nodes:
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)

    return postorder_nodes

def _is_op(operation):
    ''' If a node is op or not.
    '''
    if not isinstance(operation, Placeholder) and not isinstance(operation, Variable):
            if isinstance(operation, Operation):
                return True

    return False

def _get_ops(nodes):
    ''' Return a list of ops in a list of nodes.
    '''
    ops = []
    for eachnode in nodes:
        if _is_op(eachnode):
            ops.append(eachnode)

    return ops


class DebugSession(object):
    ''' A session to compute a particular graph.
    '''
    def __init__(self):
        ''' Session constructor.
        '''
        # Graph the session computes for.
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        ''' Context management protocal method called before `with-block`.
        '''
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Context management protocal method called after `with-block`.
        '''
        self.close()

    def delta(self, x, y):
        assert x.shape == y.shape
        if len(x.shape) == 0:
            return x - y
        #x = np.reshape(x, [np.shape(x)[0], -1])
        #y = np.reshape(y, [np.shape(y)[0], -1])
        x = x.flatten()
        y = y.flatten()
        return np.mean(np.abs(x - y), axis=0)

    def close(self):
        ''' Free all output values in nodes.
        '''
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    def run(self, operation, feed_dict=None, frameworks=['tf', 'pytorch']):
        ''' Compute the output of an operation.

        :param operation: A specific operation to be computed.
        :type operation: object of `Operation`, `Variable` or `Placeholder`.

        :param feed_dict: A mapping between placeholder and its actual value for the session.
        :type feed_dict: dict.
        '''
        # Get all prerequisite nodes using postorder traversal.
        postorder_nodes = _get_prerequisite(operation)
        postorder_ops = _get_ops(postorder_nodes)

        for each in postorder_ops:
            each.framework = frameworks[0]

        #last, this
        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value = feed_dict[node]
            else:  # Operation and variable
                node.compute_output()

        prev_output_value = operation.output_value
        prev_grad_table = compute_gradients(operation, optimizer=None)

        output_delta_dict = {}
        grad_delta_dict = {}

        for each in postorder_ops:
            output_delta_dict[each] = 0
            grad_delta_dict[each] = 0
            each.framework = frameworks[1]
            for node in postorder_nodes:
                if type(node) is Placeholder:
                    node.output_value = feed_dict[node]
                else:  # Operation and variable
                    node.compute_output()

            output_value = operation.output_value
            #print(output_value)
            grad_table = compute_gradients(operation, optimizer=None)
            for var in DEFAULT_GRAPH.trainable_variables:
                if var in grad_table:
                    grad_delta_dict[each] += self.delta(
                        grad_table[var], prev_grad_table[var])
            delta_output = self.delta(output_value, prev_output_value)
            output_delta_dict[each] = delta_output

            prev_output_value = output_value
            prev_grad_table = grad_table

        output_delta_dict = sorted(output_delta_dict.items(), key=lambda d:d[1], reverse=True)
        grad_delta_dict = sorted(grad_delta_dict.items(), key=lambda d:d[1], reverse=True)
        
        return [output_delta_dict, grad_delta_dict]

