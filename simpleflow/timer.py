import time
import os
import functools

from .operations import Operation

#Show if the timer has been started
STARTED=False

class Stats:
    #store original compute_output and compute_gradient here
    # <key=type, value=(function, function)>
    origin_func = {}
    #store propagate time for every operation
    # <key=str, value=float>
    prop_time = {}
    #store gradient time for every opertaion
    # <key=str, value=float>
    grad_time = {}

def _valid_op(op):
    '''
    Check if an op should be included in our stats,
    Normally, loss functions and normal operations should all be included
    However, if we encounter something like this
    'simpleflow.train.GradientDescentOptimizer.minimize.<locals>.MinimizationOperation'
    It should not be included since it compute the whole grad table
    '''
    if 'train' in str(op):
        return False
    return True

def _timer_wrapper(func):
    if func.__name__== "compute_gradient":
        is_grad = True
    elif func.__name__ == "compute_output":
        is_grad = False
    else:
        raise ValueError
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        op_name = args[0].__class__.__name__
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        cost_time = end_time - start_time
        if is_grad:
            if op_name in Stats.grad_time.keys():
                Stats.grad_time[op_name] += cost_time
            else:
                Stats.grad_time[op_name] = cost_time
        else:
            if op_name in Stats.prop_time.keys():
                Stats.prop_time[op_name] += cost_time
            else:
                Stats.prop_time[op_name] = cost_time
        return result
    return wrapper

def start_timer():
    '''
    Start timing for every subclass of base class Operation.
    This will wrap the function of compute_output and compute_gradient/
    '''
    global STARTED
    if STARTED:
        return
    STARTED = True
    ops = Operation.__subclasses__()
    for op in filter(_valid_op, ops):
        #print(op)
        Stats.origin_func[op] = op.compute_output, op.compute_gradient
        op.compute_output = _timer_wrapper(op.compute_output)
        op.compute_gradient = _timer_wrapper(op.compute_gradient)
    reset_timer()

def reset_timer():
    '''
    Make everything in the Stats to zero
    '''
    Stats.grad_time = {}
    Stats.prop_time = {}

def stop_timer():
    '''
    Stop timing and the the original functions will be retrieved
    '''
    global STARTED
    if not STARTED:
        return
    STARTED = False
    ops = Stats.origin_func.keys()
    for op in ops:
        op.compute_output, op.compute_gradient = Stats.origin_func[op]
    Stats.origin_func = {}

def show_timer(mode="all", form="text"):
    '''
    Output the result of the timer
    @param mode in ["all", "prop", "grad"]
    @param form in ["text", "graph"]
    '''
    if mode not in ["all", "prop", "grad"] or form not in ["text", "graph"]:
        raise ValueError
    target = Stats.prop_time
    if mode == "grad":
        target = Stats.grad_time
    elif mode == "all":
        for op in target.keys():
            if op in Stats.grad_time.keys():
                target[op] += Stats.grad_time[op]
    starget = sorted(target.items(), key=lambda d:d[1], reverse=True)
    if form == "text":
        for key,value in starget:
            print("{} \t: {}".format(key,value))
    elif form == "graph":
        import matplotlib.pyplot as plt
        plt.pie(target.values(),
                labels=target.keys(),
                autopct = '%3.1f%%',
                radius=2)
        plt.show()


__all__ = ["start_timer", "reset_timer", "stop_timer", "show_timer"]