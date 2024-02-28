"""mb.py. Copyright (C) 2024, Mukesh Dalal <mukesh.dalal@gmail.com>"""

from random import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

class ModelBox(): # main MB class
    def __init__(self):
        super().__init__()
        self.tdata = {'inputs': [], 'outputs': []}  # saved training data
        self.edata = {'inputs': [], 'outputs': []}  # saved evaluation data
        self.models = [] # list of models
        self.isproduction = [] # production status of each model
        self.functions = [] # list of functions
        self.state = 'connected' # connected, production, embedded
        self.pmaccuracy = 0.95 # accuracy threshold for production models; a non-embedded MB has production state iff at least one model has production status
        self.edata_fraction = 0.3 # fraction of cached data randomly chosen for evaluation
        self.feedback_fraction = 0.1 # fraction of calls randomly chosen for feedback from users

    def __call__(self, *xs):
        fsum = sum([f(*xs) for f in self.functions])
        msum = sum([model.predict([list(xs)]) for idx, model in enumerate(self.models) if self.isproduction[idx]])
        return (msum + fsum) / (sum(self.isproduction) + len(self.functions))
        
    def function_wrapper(self): # wrap MB  around some function f
        def decorator(f):
            def wrapper(*xs):
                self.host = f
                if self.state == 'connected':
                    y = f(*xs)
                    y = self.get_feedback(y, *xs)
                elif self.state == 'production':
                    y = (self(*xs) + f(*xs))[0] / 2
                    y = self.get_feedback(y, *xs)
                else: # self.state == 'embedded'
                    y = self(*xs)[0]
                self.add_data(y, *xs)
                return y
            return wrapper
        return decorator
    
    def sensor_wrapper(self, kind='direct'): # wrap a sensor around some function g
        def decorator(g): # add other sensors as needed
            def direct(*xs): # direct sensor
                y = g(*xs)
                self.add_data(y, *xs)
                return y
            def inverse(*yxs): # inverse sensor, yxs = (y, xs except xlast)
                xlast = g(*yxs)
                self.add_data(yxs[0], *yxs[0:], xlast)
                return xlast
            return eval(kind)
        assert kind in ['direct', 'inverse'], "sensor kind must be 'direct' or 'inverse'"
        return decorator
    
    def add_data(self, y, *xs):
        if random() <= self.edata_fraction: # cache as evaluation data
            self.edata['inputs'].append(list(xs))
            self.edata['outputs'].append(y)
        else:
            self.tdata['inputs'].append(list(xs))
            self.tdata['outputs'].append(y)
    
    def add_model(self, model):
        self.models.append(model)
        self.isproduction.append(False)
        self.print('After adding a model, but before training it')
        return self.train(len(self.models) - 1)

    def remove_model(self, idx):
        self.models.pop(idx)
        self.isproduction.pop(idx)

    def train(self, idx):
        self.models[idx].fit(self.tdata['inputs'], self.tdata['outputs'])
        return self.test(idx)

    def test(self, idx):
        if self.edata['inputs'] == []:
            return False
        score = self.models[idx].score(self.edata['inputs'], self.edata['outputs'])
        res = score >= self.pmaccuracy
        print(f"Compare Model {idx} score = {score} with pmaccuracy {self.pmaccuracy} => production = {res}")
        self.isproduction[idx] = res
        production = sum(self.isproduction)
        if production == 0: 
            self.state = 'connected'
        elif self.state == 'connected': 
            self.state = 'production'
        return res

    def train_all(self):
        for idx in range(len(self.models)):
            self.train(idx)

    def test_all(self):
        for idx in range(len(self.models)):
            self.test(idx)

    def productionize(self):
        self.state = 'production' 

    def embed(self):
        self.functions.append(self.host)
        self.state = 'embedded' 

    def add_dataset(self, x, y, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self.'+ kind)['inputs'] += x
        eval('self.'+ kind)['outputs'] += y

    def remove_dataset(self, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        eval('self.'+ kind)['inputs'] = []
        eval('self.'+ kind)['outputs'] = []

    def add_function(self, fn):
        self.functions.append(fn)

    def remove_function(self, idx):
        self.functions.pop(idx)

    def get_feedback(self, y, *xs):
        if random() <= self.feedback_fraction:
            newy = input(f"x={roundl(list(xs))}, {y=:.1f}. To override y, type a new value (float) and return, otherwise just press return:")
            if newy != '':  
                return float(newy)
        return y 

    def print(self, tag, ntail=2): # print tag string and then MB with ntail inputs and outputs
        print(f"{tag}: state:{self.state}, #functions:{len(self.functions)}, #models:{len(self.models)}, production:{self.isproduction}, #tdata:{len(self.tdata['inputs'])}, #edata:{len(self.edata['inputs'])}; tinputs:{'...' if len(self.tdata['inputs']) > ntail  else ''}{roundl(self.tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self.tdata['outputs']) > ntail else ''}{roundl(self.tdata['outputs'][-ntail:])}; einputs:{'...' if len(self.edata['inputs']) > ntail  else ''}{roundl(self.edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self.edata['outputs']) > ntail else ''}{roundl(self.edata['outputs'][-ntail:])})")

def generate_data(f, count=1000):
    return [[x, x+1] for x in range(count)], [f(x, x+1) for x in range(count)]

def repeat_function(f, count=10, scale=100):
    for _ in range(count):
        f(random()*scale, random()*scale)

def roundl(xl, places=0):
    if type(xl) in (list, tuple): 
        return [roundl(x, places) for x in xl]
    return round(xl, places)

mb = ModelBox()
@mb.function_wrapper()
def f(x0, x1): 
    return 3 * x0 + x1

mb.print('Intial MB')
repeat_function(f)
mb.print('After a few function calls')

mb.add_model(LinearRegression())
mb.print('After training and testing the added model')
repeat_function(f)
mb.print('After a few more function calls')

if mb.state == 'production': 
    mb.embed()
    mb.print('After embedding')
    repeat_function(f)
    mb.print('After a few more function calls')

mb.add_model(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mb.print('After training and testing the added model')
repeat_function(f)
mb.print('After a few more function calls')

if mb.state == 'embedded':
    xy = generate_data(f)
    mb.add_dataset(xy[0], xy[1])
    mb.print('After adding more data but before training')
    mb.train_all()
    mb.print('After training and testing with the new data')
    repeat_function(f)
    mb.print('After a few more function calls')

if mb.isproduction[1] == False:
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

mb.remove_function(1)
mb.print('After removing the second function')

mb.remove_dataset()
mb.print('After removing all data')
repeat_function(f)
mb.print('After a few more function calls')

@mb.sensor_wrapper()
def fcopy(x0, x1):  # y
    return 3 * x0 + x1

repeat_function(fcopy)
mb.print('After a few direct-sensor calls')

@mb.sensor_wrapper('inverse')
def finv(y, x0):  # x1
    return y - 3 * x0

repeat_function(finv)
mb.print('After a few inverse-sensor calls')