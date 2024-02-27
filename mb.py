"""mb.py. Copyright (C) 2024, Mukesh Dalal <mukesh.dalal@gmail.com>"""

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore") 

mb = {  'data': {'inputs': [], 'outputs': []},
        'models': [],
        'production': [],
        'functions': [],
        'maf': lambda x: (msum(mb, x) + fsum(mb, x)) / (sum(mb['production']) + len(mb['functions'])),
        'state': 'connected', # connected, production, embedded
        'pthreshold': 0.95,
    }
    
def mbw(mb):
    def decorator(f):
        def wrapper(x):
            mb['host'] = f
            if mb['state'] == 'connected':
                y = f(x)
                y = feedback(x, y)
            elif mb['state'] == 'production':
                y = (mb['maf'](x) + f(x))[0] / 2
                y = feedback(x, y)
            else: # mb['state'] == 'embedded'
                y = mb['maf'](x)[0]
            mb['data']['inputs'].append([x])
            mb['data']['outputs'].append(y)
            return y
        return wrapper
    return decorator

def fsum(mb, x):
    return sum([f(x) for f in mb['functions']])

def msum(mb, x):
    return sum([model.predict([[x]]) for idx, model in enumerate(mb['models']) if mb['production'][idx]])

def feedback(x, y):
    newy = input(f"{x=}, {y=}. To override y, type a new value (float) and return, otherwise just press return:")
    if newy != '':  
        return float(newy)
    return y  

def addmodel(mb, model):
    mb['models'].append(model)
    mb['production'].append(False)
    mbprint('After adding a model, but before training it', mb)
    return train(mb, len(mb['models']) - 1)

def removemodel(mb, idx):
    mb['models'].pop(idx)
    mb['production'].pop(idx)

def train(mb, idx):
    mb['models'][idx].fit(mb['data']['inputs'], mb['data']['outputs'])
    return test(mb, idx)

def test(mb, idx):
    score = mb['models'][idx].score(mb['data']['inputs'], mb['data']['outputs'])
    res = score >= mb['pthreshold']
    print(f"Compare Model {idx} score = {score} with pthreshold {mb['pthreshold']} => production = {res}")
    mb['production'][idx] = res
    production = sum(mb['production'])
    if production == 0: 
        mb['state'] = 'connected'
    elif mb['state'] == 'connected': 
        mb['state'] = 'production'
    return res

def trainall(mb):
    for idx in range(len(mb['models'])):
        train(mb, idx)

def testall(mb):
    for idx in range(len(mb['models'])):
        test(mb, idx)

def productionize(mb):
    mb['state'] = 'production' 

def embed(mb):
    mb['functions'].append(mb['host'])
    mb['state'] = 'embedded' 

def adddata(mb, x, y):
    mb['data']['inputs'] += x
    mb['data']['outputs'] += y

def removedata(mb):
    mb['data']['inputs'] = []
    mb['data']['outputs'] = []

def addfn(mb, fn):
    mb['functions'].append(fn)

def removefn(mb, idx):
    mb['functions'].pop(idx)

def mbprint(tag, mb):
    print(f"{tag}: state:{mb['state']}, #functions:{len(mb['functions'])}, #models:{len(mb['models'])}, production:{mb['production']}, pthreshold:{mb['pthreshold']}; inputs:{'...' if len(mb['data']['inputs']) > 10 else ''}{mb['data']['inputs'][-10:]}; outputs:{'...' if len(mb['data']['outputs']) > 10 else ''}{mb['data']['outputs'][-10:]}")

def gendata(f, count=1000):
    return [[x] for x in range(count)], [f(x) for x in range(count)]

@mbw(mb)
def f(x): 
    return 3 * x + 2

mbprint('Intial MB', mb)
f(1)
f(2)
mbprint('After a few function calls',mb)

addmodel(mb, LinearRegression())
mbprint('After training and testing the added model', mb)
f(3)
f(4)
mbprint('After a few more function calls', mb)

if mb['state'] == 'production': 
    embed(mb)
    mbprint('After embedding', mb)
    f(5)
    f(6)
    mbprint('After a few more function calls', mb)

addmodel(mb, MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mbprint('After training and testing the added model', mb)
f(7)
f(8)
mbprint('After a few more function calls', mb)

xy = gendata(f)
adddata(mb, xy[0], xy[1])
mbprint('After adding more data but before training', mb)
trainall(mb)
mbprint('After training and testing with the new data', mb)
f(998)
f(999)
mbprint('After a few more function calls', mb)

if mb['production'][1] == False:
    mb['pthreshold'] = -100
    mbprint('After updating threshold', mb)
    testall(mb)
    mbprint('After retesting all models with the new threshold', mb)
    f(998)
    f(999)
    mbprint('After a few more function calls', mb)

removemodel(mb, 1)
mb['pthreshold'] = 0.95
mbprint('After removing the second model and reverting the threshold', mb)

addfn(mb, lambda x: x * x)
mbprint('After adding a new function', mb)

removefn(mb, 1)
mbprint('After removing the second function', mb)

removedata(mb)
mbprint('After removing all data', mb)
f(998)
f(999)
mbprint('After a few more function calls', mb)