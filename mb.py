"""mb.py. Copyright (C) 2024, Mukesh Dalal <mukesh.dalal@gmail.com>"""

import inspect
from random import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

class FloatCallback(float): # float with a callback function
    def __new__(cls, value, callback):
        obj = super().__new__(cls, value)
        obj.function = callback
        return obj
    def callback(self, *args):
        return self.function(*args)
    
class ModelBox(): # main MB class
    def __init__(self):
        super().__init__()
        self._tdata = {'inputs': [], 'outputs': []}  # saved training data
        self._edata = {'inputs': [], 'outputs': []}  # saved evaluation data
        self._models = [] # list of models
        self._isproduction = [] # production status of each model
        self._functions = [] # list of functions
        self._state = 'connected' # connected, production, embedded
        self.pmaccuracy = 0.95 # accuracy threshold for production models; a non-embedded MB has production state iff at least one model has production status
        self.edata_fraction = 0.3 # fraction of cached data randomly chosen for evaluation
        self.feedback_fraction = 0.1 # fraction of calls randomly chosen for feedback from users
        
    def __call__(self, *xs, cvals=None):
        if cvals == None:
            cvals = tuple()
        fsum = sum([f(*xs) for f in self._functions])
        msum = sum([model.predict([list(xs)+list(cvals)]) for idx, model in enumerate(self._models) if self._isproduction[idx]])
        return (msum + fsum) / (sum(self._isproduction) + len(self._functions))
        
    def function_wrapper(self, *cfields): # wrap MB around some function f with cvals data from cfields
        def decorator(f):
            def wrapper(*xs):
                self.host = f
                cvals = self._cvals(*cfields)
                if self._state == 'connected':
                    y = f(*xs)
                    y = self._get_feedback(y, *xs, cvals=cvals)
                elif self._state == 'production':
                    y = (self(*xs, cvals=cvals) + f(*xs))[0] / 2
                    y = self._get_feedback(y, *xs, cvals=cvals)
                else: # self._state == 'embedded'
                    y = self(*xs, cvals=cvals)[0]
                kind = self._add_data(y, *xs, *cvals)
                return FloatCallback(y, self._callback(kind, *xs, *cvals))
            return wrapper
        return decorator   

    def sensor_wrapper(self, kind='direct'): # wrap a sensor around some function g
        def decorator(g): # add other sensors as needed
            def direct(*xs): # direct sensor
                y = g(*xs)
                self._add_data(y, *xs)
                return y
            def inverse(*yxs): # inverse sensor, yxs = (y, xs except xfirst)
                xfirst = g(*yxs)
                self._add_data(yxs[0], xfirst, *yxs[1:])
                return xfirst
            return eval(kind)
        assert kind in ['direct', 'inverse'], "sensor kind must be 'direct' or 'inverse'"
        return decorator
    
    def add_model(self, model):
        self._models.append(model)
        self._isproduction.append(False)
        self.print('After adding a model, but before training it')
        return self.train(len(self._models) - 1)

    def remove_model(self, idx):
        self._models.pop(idx)
        self._isproduction.pop(idx)

    def train(self, idx):
        self._models[idx].fit(self._tdata['inputs'], self._tdata['outputs'])
        return self.test(idx)

    def test(self, idx):
        if self._edata['inputs'] == []:
            return False
        score = self._models[idx].score(self._edata['inputs'], self._edata['outputs'])
        res = score >= self.pmaccuracy
        print(f"Compare Model {idx} score = {score} with pmaccuracy {self.pmaccuracy} => production = {res}")
        self._isproduction[idx] = res
        production = sum(self._isproduction)
        if production == 0: 
            self._state = 'connected'
        elif self._state == 'connected': 
            self._state = 'production'
        return res

    def train_all(self):
        for idx in range(len(self._models)):
            self.train(idx)

    def test_all(self):
        for idx in range(len(self._models)):
            self.test(idx)

    def embed(self):
        self._functions.append(self.host)
        self._state = 'embedded' 

    def add_dataset(self, x, y, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self._'+ kind)['inputs'] += x
        eval('self._'+ kind)['outputs'] += y

    def remove_dataset(self, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self._'+ kind)['inputs'] = []
        eval('self._'+ kind)['outputs'] = []

    def add_function(self, fn):
        self._functions.append(fn)

    def remove_function(self, idx):
        self._functions.pop(idx)

    def _get_feedback(self, y, *xs, cvals=None):
        if cvals == None:
            cvals = tuple()
        if random() <= self.feedback_fraction:
            newy = input(f"x={roundl(list(xs))}, context={roundl(list(cvals))}, {y=:.1f}. To override y, type a new value (float) and return, otherwise just press return:")
            if newy != '':  
                return float(newy)
        return y 

    def print(self, tag, ntail=2): # print tag string and then MB with ntail inputs and outputs
        print(f"{tag}: state:{self._state}, #functions:{len(self._functions)}, #models:{len(self._models)}, production:{self._isproduction}, #tdata:{len(self._tdata['inputs'])}, #edata:{len(self._edata['inputs'])}; tinputs:{'...' if len(self._tdata['inputs']) > ntail  else ''}{roundl(self._tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self._tdata['outputs']) > ntail else ''}{roundl(self._tdata['outputs'][-ntail:])}; einputs:{'...' if len(self._edata['inputs']) > ntail  else ''}{roundl(self._edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self._edata['outputs']) > ntail else ''}{roundl(self._edata['outputs'][-ntail:])})")
     
    def _add_data(self, y, *xs):
        if random() <= self.edata_fraction: # cache as evaluation data
            self._edata['inputs'].append(list(xs))
            self._edata['outputs'].append(y)
            return 'edata'
        else:
            self._tdata['inputs'].append(list(xs))
            self._tdata['outputs'].append(y)
            return 'tdata'
                   
    def _callback(self, kind, *args):
        def inner(y):
            nonlocal self, kind, args
            idx = eval('self._'+ kind)['inputs'].index(list(args))
            eval('self._'+ kind)['outputs'][idx] = y
            return y
        return inner
    
    @staticmethod
    def _cvals(*cfields):
        try:
            frame = inspect.currentframe().f_back
            all_variables = {**frame.f_globals, **frame.f_locals}
            cvals = [v for k, v in all_variables.items() if k in cfields]
            return cvals
        finally:
            del frame

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
@mb.function_wrapper('globalx')
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

if mb._state == 'production': 
    mb.embed()
    mb.print('After embedding')
    repeat_function(f)
    mb.print('After a few more function calls')

mb.add_model(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mb.print('After training and testing the added model')
repeat_function(f)
mb.print('After a few more function calls')

if mb._state == 'embedded':
    xy = generate_data(f)
    mb.add_dataset(xy[0], xy[1])
    mb.print('After adding more data but before training')
    mb.train_all()
    mb.print('After training and testing with the new data')
    repeat_function(f)
    mb.print('After a few more function calls')

if mb._isproduction[1] == False:
    mb.pmaccuracy = -100
    mb.print('After updating threshold')
    mb.test_all()
    mb.print('After retesting all models with the new threshold')
    repeat_function(f)
    mb.print('After a few more function calls')

mb.remove_model(1)
mb.pmaccuracy = 0.95
mb.print('After removing the second model and reverting the threshold')

mb.add_function(lambda x: x * x)
mb.print('After adding a new function')

mb.remove_function(-1)
mb.print('After removing the last function')

mb.remove_dataset()
mb.print('After removing all data')
repeat_function(f)
mb.print('After a few more function calls')

@mb.sensor_wrapper()
def fcopy(x0, x1, x3):  # y
    return 3 * x0 + x1 + x3

repeat_function(fcopy, arity=3)
mb.print('After a few direct-sensor calls')

@mb.sensor_wrapper('inverse')
def finv(y, x1, x2):  # x1
    return (y - x1 -  x2) / 3

repeat_function(finv, arity=3)
mb.print('After a few inverse-sensor calls')

globalx = 1
y = f(2, 3)
y.callback(100.0)
for kind in ('tdata', 'edata'):
    try:
        idx = eval('mb._'+ kind)['inputs'].index([2, 3, 1])
        print(f"Feedback callback: {y:.1f} updated to {eval('mb._'+ kind)['outputs'][idx]} in mb._{kind}['outputs'] for inputs [2, 3, 1]")
    except: pass