"""mb.py. Copyright (C) 2024, Mukesh Dalal <mukesh.dalal@gmail.com>"""

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore") 

class ModelBox():
    def __init__(self):
        super().__init__()
        self.data = {'inputs': [], 'outputs': []}
        self.models = []
        self.production = []
        self.functions = []
        self.state = 'connected' # connected, production, embedded
        self.pthreshold = 0.95

    def __call__(self, x):
        fsum = sum([f(x) for f in self.functions])
        msum = sum([model.predict([[x]]) for idx, model in enumerate(self.models) if self.production[idx]])
        return (msum + fsum) / (sum(self.production) + len(self.functions))
        
    def mbw(self):
        def decorator(f):
            def wrapper(x):
                self.host = f
                if self.state == 'connected':
                    y = f(x)
                    y = self.feedback(x, y)
                elif self.state == 'production':
                    y = (self(x) + f(x))[0] / 2
                    y = self.feedback(x, y)
                else: # self.state == 'embedded'
                    y = self(x)[0]
                self.data['inputs'].append([x])
                self.data['outputs'].append(y)
                return y
            return wrapper
        return decorator
    
    def addmodel(self, model):
        self.models.append(model)
        self.production.append(False)
        self.print('After adding a model, but before training it')
        return self.train(len(self.models) - 1)

    def removemodel(self, idx):
        self.models.pop(idx)
        self.production.pop(idx)

    def train(self, idx):
        self.models[idx].fit(self.data['inputs'], self.data['outputs'])
        return self.test(idx)

    def test(self, idx):
        score = self.models[idx].score(self.data['inputs'], self.data['outputs'])
        res = score >= self.pthreshold
        print(f"Compare Model {idx} score = {score} with pthreshold {self.pthreshold} => production = {res}")
        self.production[idx] = res
        production = sum(self.production)
        if production == 0: 
            self.state = 'connected'
        elif self.state == 'connected': 
            self.state = 'production'
        return res

    def trainall(self):
        for idx in range(len(self.models)):
            self.train(idx)

    def testall(self):
        for idx in range(len(self.models)):
            self.test(idx)

    def productionize(self):
        self.state = 'production' 

    def embed(self):
        self.functions.append(self.host)
        self.state = 'embedded' 

    def adddata(self, x, y):
        self.data['inputs'] += x
        self.data['outputs'] += y

    def removedata(self):
        self.data['inputs'] = []
        self.data['outputs'] = []

    def addfn(self, fn):
        self.functions.append(fn)

    def removefn(self, idx):
        self.functions.pop(idx)

    def feedback(self, x, y):
        newy = input(f"{x=}, {y=}. To override y, type a new value (float) and return, otherwise just press return:")
        if newy != '':  
            return float(newy)
        return y  

    def print(self, tag):
        print(f"{tag}: state:{self.state}, #functions:{len(self.functions)}, #models:{len(self.models)}, production:{self.production}, pthreshold:{self.pthreshold}; inputs:{'...' if len(self.data['inputs']) > 10 else ''}{self.data['inputs'][-10:]}; outputs:{'...' if len(self.data['outputs']) > 10 else ''}{self.data['outputs'][-10:]}")

def gendata(f, count=1000):
    return [[x] for x in range(count)], [f(x) for x in range(count)]

mb = ModelBox()
@mb.mbw()
def f(x): 
    return 3 * x + 2

mb.print('Intial MB')
f(1)
f(2)
mb.print('After a few function calls')

mb.addmodel(LinearRegression())
mb.print('After training and testing the added model')
f(3)
f(4)
mb.print('After a few more function calls')

if mb.state == 'production': 
    mb.embed()
    mb.print('After embedding')
    f(5)
    f(6)
    mb.print('After a few more function calls')

mb.addmodel(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mb.print('After training and testing the added model')
f(7)
f(8)
mb.print('After a few more function calls')

xy = gendata(f)
mb.adddata(xy[0], xy[1])
mb.print('After adding more data but before training')
mb.trainall()
mb.print('After training and testing with the new data')
f(998)
f(999)
mb.print('After a few more function calls')

if mb.production[1] == False:
    mb.pthreshold = -100
    mb.print('After updating threshold')
    mb.testall()
    mb.print('After retesting all models with the new threshold')
    f(998)
    f(999)
    mb.print('After a few more function calls')

mb.removemodel(1)
mb.pthreshold = 0.95
mb.print('After removing the second model and reverting the threshold')

mb.addfn(lambda x: x * x)
mb.print('After adding a new function')

mb.removefn(1)
mb.print('After removing the second function')

mb.removedata()
mb.print('After removing all data')
f(998)
f(999)
mb.print('After a few more function calls')