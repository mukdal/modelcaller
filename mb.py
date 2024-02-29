"""mb.py. Copyright (C) 2024, Mukesh Dalal <mukesh.dalal@gmail.com>"""

import inspect
from random import random
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

class CallbackBase: # for adding callbacks to objects w/o dynamic attributes
    def __init__(self, value, callback):
        self.value = value
        self.cbfn = callback
    def callback(self, *args, **kwargs):
        return self.cbfn(*args, **kwargs)
    def __getattr__(self, item): # delegate any attribute not defined here to the value
        return getattr(self.value, item)
    
class FloatCallback(CallbackBase, float): # for floats (and ints)
    def __new__(cls, value, callback):
        assert isinstance(value, float) or isinstance(value, int), "non-float/int value passed to FloatCallback"
        obj = float.__new__(cls, value)  # Create a float instance
        obj.callback = callback  # Assign the callback directly to this float instance
        return obj

@dataclass
class MBConfig: # default MB configuration
    auto_cache = True # auto cache host call data?
    auto_test = True # auto test after training a model?
    auto_train = True # auto train after adding a model?
    edata_fraction = 0.3 # fraction of data cached for evaluation (instead of training)
    feedback_fraction = 0.1 # fraction of host calls randomly chosen for feedback
    qlty_threshold = 0.95 # quality (accuracy) threshold for models

class ModelBox(): # main MB class
    def __init__(self, mbconfig=MBConfig()):
        super().__init__()
        self.auto_cache = mbconfig.auto_cache
        self.auto_test = mbconfig.auto_test
        self.auto_train = mbconfig.auto_train
        self.edata_fraction = mbconfig.edata_fraction 
        self.feedback_fraction = mbconfig.feedback_fraction 
        self.qlty_threshold =  mbconfig.qlty_threshold 
        self._call_state = 'host' # host, MB, both : who to call when the host fn is called?
        self._edata = {'inputs': [], 'outputs': []}  # saved evaluation data
        self._functions = [] # list of functions
        self._host = None   # original unwrapped host function (gets populated by function_wrap)
        self._models = [] # list of models
        self._tdata = {'inputs': [], 'outputs': []}  # saved training data
        
    def __call__(self, *xs, cvals=None):
        if cvals == None:
            cvals = tuple()
        fsum = sum([f(*xs) for f in self._functions])
        msum = sum([model.predict([list(xs)+list(cvals)]) for idx, model in enumerate(self._models) if model.quality])
        return (msum + fsum) / (self._count_quality_models() + len(self._functions))
    
    def add_dataset(self, x, y, kind='tdata'): # training (default)or evaluation
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self._'+ kind)['inputs'] += x
        eval('self._'+ kind)['outputs'] += y
    
    def add_function(self, fn=None):
        if fn == None: # add the original unwrapped host function
            fn = self._host
        self._functions.append(fn)
        return len(self._functions) - 1 # return index of the added function
    
    def add_model(self, model):
        self._models.append(model)
        model.quality = False
        self.print('After adding a model, but before training it')
        idx = len(self._models) - 1
        if self.auto_train:
            self.train(idx)
        return idx

    def find_data(self, x, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        try:
            idx = eval('self._'+ kind)['inputs'].index(x)
            y = eval('self._'+ kind)['outputs'][idx]
        except:
            idx = -1
            y = None
        return idx, y
    
    def function_wrap(self, *cfields): # wrap MB around some function f with cvals data from cfields
        def decorator(f):
            def wrapper(*xs):
                self._host = f
                cvals = self._cvals(*cfields)
                if self._call_state == 'host':
                    y = f(*xs)  # call only host
                    y = self._get_feedback(y, *xs, cvals=cvals)
                elif self._call_state == 'MB': # no feedback
                    y = self(*xs, cvals=cvals)[0] # call only MB
                else: # self._call_state == 'both': 
                    y = (self(*xs, cvals=cvals) + f(*xs))[0] / 2 # call both host and MB
                    y = self._get_feedback(y, *xs, cvals=cvals)
                if self.auto_cache:
                    kind = self._add_data(y, *xs, *cvals)
                    y = FloatCallback(y, self._callback(kind, *xs, *cvals)) # y must be int or float
                return y 
            return wrapper
        return decorator   
    
    def get_call_state(self):
        return self._call_state
    
    def get_model_quality(self, idx):
        return self._models[idx].quality
    
    def print(self, tag, ntail=2): # print tag string and then MB with ntail inputs and outputs
        print(f"{tag}: state:{self._call_state}, #functions:{len(self._functions)}, model-qualities:{[m.quality for m in self._models]}, #tdata:{len(self._tdata['inputs'])}, #edata:{len(self._edata['inputs'])}; tinputs:{'...' if len(self._tdata['inputs']) > ntail  else ''}{roundl(self._tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self._tdata['outputs']) > ntail else ''}{roundl(self._tdata['outputs'][-ntail:])}; einputs:{'...' if len(self._edata['inputs']) > ntail  else ''}{roundl(self._edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self._edata['outputs']) > ntail else ''}{roundl(self._edata['outputs'][-ntail:])})")
    
    def remove_data(self, idx, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self._'+ kind)['inputs'].pop(idx)
        eval('self._'+ kind)['outputs'].pop(idx)
        
    def remove_dataset(self, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self._'+ kind)['inputs'] = []
        eval('self._'+ kind)['outputs'] = []
    
    def remove_function(self, idx):
        self._functions.pop(idx)
    
    def remove_model(self, idx):
        self._models.pop(idx)

    def sensor_wrap(self, skind='direct'): # wrap a sensor around some function g
        def decorator(g): # add other sensors as needed
            def direct(*xs): # direct sensor
                y = g(*xs)
                self._add_data(y, *xs)
                return y
            def inverse(*yxs): # inverse sensor, yxs = (y, xs except xfirst)
                xfirst = g(*yxs)
                self._add_data(yxs[0], xfirst, *yxs[1:])
                return xfirst
            return eval(skind)  # correct kind of sensor
        assert skind in ['direct', 'inverse'], "sensor kind must be 'direct' or 'inverse'"
        return decorator
    
    def set_call_state(self, newstate):
        assert newstate in ['host', 'both', 'MB'], "call_state must be 'host', 'both' or 'MB'"
        self._call_state = newstate 
    
    def test(self, idx, all=False):
        if self._edata['inputs'] == []:
            return False
        model = self._models[idx]
        score = model.score(self._edata['inputs'], self._edata['outputs'])
        res = score >= self.qlty_threshold
        print(f"Compare Model {idx} score = {score} with qlty_threshold {self.qlty_threshold} => quality = {res}")
        model.quality = res
        if not all:  # not testing all models
            self._auto_call_state()
        return res
    
    def test_all(self):
        for idx in range(len(self._models)):
            self.test(idx, all=True)
        self._auto_call_state()

    def train(self, idx, all=False):
        self._models[idx].fit(self._tdata['inputs'], self._tdata['outputs'])
        if not all and self.auto_test:
            self.test(idx)

    def train_all(self):
        for idx in range(len(self._models)):
            self.train(idx, all=True)
        if self.auto_test:
            self.test_all()
     
    def _add_data(self, y, *xs):
        if random() <= self.edata_fraction: # save as evaluation data
            self._edata['inputs'].append(list(xs))
            self._edata['outputs'].append(y)
            return 'edata'
        self._tdata['inputs'].append(list(xs))
        self._tdata['outputs'].append(y)
        return 'tdata'
    
    def _auto_call_state(self):
        if self._count_quality_models() == 0: 
            self._call_state = 'host'
        elif self._call_state == 'host': 
            self._call_state = 'both'
                         
    def _callback(self, kind, *args):
        def inner(y):
            nonlocal self, kind, args
            idx = eval('self._'+ kind)['inputs'].index(list(args))
            eval('self._'+ kind)['outputs'][idx] = y
            return y
        return inner
    
    def _count_quality_models(self):
        return sum([m.quality for m in self._models])
 
    @staticmethod
    def _cvals(*cfields): # get cfield values from the context
        try:
            frame = inspect.currentframe().f_back
            all_variables = {**frame.f_globals, **frame.f_locals}
            cvals = [v for k, v in all_variables.items() if k in cfields]
            return cvals
        finally:
            del frame
         
    def _get_feedback(self, y, *xs, cvals=None):
        if cvals == None:
            cvals = tuple()
        if random() <= self.feedback_fraction:
            newy = input(f"x={roundl(list(xs))}, context={roundl(list(cvals))}, {y=:.1f}. To override y, type a new value (float) and return, otherwise just press return:")
            if newy != '':  
                return float(newy)
        return y     

def generate_data(f, count=1000, scale=100):
    global globalx
    inputs, outputs = [], []
    for _ in range(count):
        globalx = random() * scale
        x0 = random() * scale
        x1 = random() * scale
        inputs.append([x0, x1, globalx])
        outputs.append(f(x0, x1))
    return inputs, outputs

def repeat_function(f, arity=2, count=10, scale=100):
    global globalx
    for _ in range(count):
        globalx = random() * scale
        args = [random() * scale for _ in range(arity)]
        f(*args)

def roundl(xl, places=0):
    if type(xl) in (list, tuple): 
        return [roundl(x, places) for x in xl]
    return round(xl, places)

mb = ModelBox()
@mb.function_wrap('globalx')
def f(x0, x1): 
    global globalx
    return 3 * x0 + x1 + globalx

mb.print('Intial MB')
repeat_function(f)
mb.print('After a few function calls')

mb.add_model(LinearRegression())
mb.print('After training and testing the added model')
repeat_function(f)
mb.print('After a few more function calls')

if mb.get_call_state() == 'both': 
    mb.set_call_state('MB')
    mb.print('After embedding')
    repeat_function(f)
    mb.print('After a few more function calls')

midx = mb.add_model(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mb.print('After training and testing the added model')
repeat_function(f)
mb.print('After a few more function calls')

if mb.get_call_state() == 'MB':
    xy = generate_data(f)
    mb.add_dataset(xy[0], xy[1])
    mb.print('After adding more data but before training')
    mb.train_all()
    mb.print('After training and testing with the new data')
    repeat_function(f)
    mb.print('After a few more function calls')

if mb.get_model_quality(midx) == False:
    mb.qlty_threshold = -100
    mb.print('After updating threshold')
    mb.test_all()
    mb.print('After retesting all models with the new threshold')
    repeat_function(f)
    mb.print('After a few more function calls')

mb.remove_model(1)
mb.qlty_threshold = 0.95
mb.print('After removing the second model and reverting the threshold')

mb.add_function(lambda x: x * x)
mb.print('After adding a new function')

mb.remove_function(-1)
mb.print('After removing the last function')

mb.remove_dataset()
mb.print('After removing all data')
repeat_function(f)
mb.print('After a few more function calls')

@mb.sensor_wrap()
def fcopy(x0, x1, x3):  # y
    return 3 * x0 + x1 + x3

repeat_function(fcopy, arity=3)
mb.print('After a few direct-sensor calls')

@mb.sensor_wrap('inverse')
def finv(y, x1, x2):  # x1
    return (y - x1 -  x2) / 3

repeat_function(finv, arity=3)
mb.print('After a few inverse-sensor calls')

globalx = 1
y = f(2, 3)
y.callback(100.0)
for kind in ('tdata', 'edata'):
    idx, out = mb.find_data([2, 3, 1], kind)
    if idx >= 0:
        print(f"Feedback callback: {y:.1f} updated to {out} in _{kind}['outputs'][{idx}] for inputs [2, 3, 1]")