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

def fsum(mb, x):
    return sum([f(x) for f in mb['functions']])

def msum(mb, x):
    return sum([model.predict([[x]]) for idx, model in enumerate(mb['models']) if mb['production'][idx]])

def mbw(mb):
    def decorator(f):
        def wrapper(input):
            mb['host'] = f
            if mb['state'] == 'connected':
                output = f(input)
            elif mb['state'] == 'production':
                output = (mb['maf'](input) + f(input))[0] / 2
            else: # mb['state'] == 'embedded'
                output = mb['maf'](input)[0]
            mb['data']['inputs'].append([input])
            mb['data']['outputs'].append(output)
            return output
        return wrapper
    return decorator

def addmodel(mb, model):
    mb['models'].append(model)
    mb['production'].append(False)
    mbprint('After adding a model, but before training it', mb)
    return train(mb, len(mb['models']) - 1)

def train(mb, idx):
    mb['models'][idx].fit(mb['data']['inputs'], mb['data']['outputs'])
    return test(mb, idx)

def test(mb, idx):
    score = mb['models'][idx].score(mb['data']['inputs'], mb['data']['outputs'])
    res = score >= mb['pthreshold']
    print(f"Model {idx} score = {score} meets pthreshold {mb['pthreshold']} => production = {res}")
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

def productionize(mb):
    mb['state'] = 'production' 

def embed(mb):
    mb['functions'].append(mb['host'])
    mb['state'] = 'embedded' 

def mbprint(tag, mb):
    print(f"{tag}: state:{mb['state']}, #models:{len(mb['models'])}, production:{mb['production']}, #functions:{len(mb['functions'])}, pthreshold:{mb['pthreshold']}, inputs:{mb['data']['inputs']}, outputs:{mb['data']['outputs']}")

@mbw(mb)
def f(x): 
    return 3 * x + 2

mbprint('Intial MB', mb)
f(1); f(2); mbprint('After a few function calls',mb)

addmodel(mb, LinearRegression())
mbprint('After training and testing the added model', mb)
f(3); f(4); mbprint('After a few more function calls', mb)

if mb['state'] == 'production': 
    embed(mb)
    mbprint('After embedding', mb)
    f(5); f(6); mbprint('After a few more function calls', mb)

addmodel(mb, MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mbprint('After training and testing the added model', mb)
f(7); f(8); mbprint('After a few more function calls', mb)

mb['pthreshold'] = -100
mbprint('After updating threshold', mb)
trainall(mb)
mbprint('After retraining all models with the new threshold', mb)
f(9); f(10); mbprint('After a few more function calls', mb)