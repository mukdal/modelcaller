import random
import numpy as np
from inspect import currentframe
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")
from modelcaller import ModelCaller, MCconfig, mc_wrapd, mc_wrap
random.seed(42)

def generate_data(f, count=1000, scale=100):
    global globalx
    inputs = np.zeros((count, 3))
    outputs = np.zeros(count)
    for i in range(count):
        globalx = random.random() * scale
        x0 = random.random() * scale
        x1 = random.random() * scale
        inputs[i] = [x0, x1, globalx]
        outputs[i] = f(x0, x1)
    return inputs, outputs

def repeat_function(f, arity=2, count=10, scale=100):
    global globalx
    for _ in range(count):
        globalx = random.random() * scale
        args = [random.random() * scale for _ in range(arity)]
        f(*args)

@mc_wrapd(cargs=['globalx'])
def f(x0, x1): 
    global globalx
    return 3 * x0 + x1 + globalx
mc = f._mc

mc.print('A new wrapped MC with one context argument', full=True)
repeat_function(f)
mc.print('After a few function calls')

mc.add_model(LinearRegression())
mc.print('After training and testing the added model')
repeat_function(f)
mc.print('After a few more function calls')

if mc.get_call_target() == 'both': 
    mc.merge_host()
    mc.print('After merging host function')
    repeat_function(f)
    mc.print('After a few more function calls')

midx = mc.add_model(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
mc.print('After training and testing the added model')
repeat_function(f)
mc.print('After a few more function calls')

if mc.get_call_target() == 'MC':
    xy = generate_data(mc.get_host())
    mc.add_dataset(xy[0], xy[1])
    mc.print('After adding more data but before training')
    mc.train_all()
    mc.print('After training and testing with the new data')
    repeat_function(f)
    mc.print('After a few more function calls')

if mc.get_model_quality(midx) == False:
    mc.qlty_threshold = -100
    mc.print('After updating qlty_threshold', full=True)
    mc.test_all()
    mc.print('After retesting all models with the new threshold')
    repeat_function(f)
    mc.print('After a few more function calls')

mc.remove_model(1)
mc.qlty_threshold = 0.95
mc.print('After removing the second model and reverting the threshold', full=True)

mc.add_function(lambda x: x * x)
mc.print('After adding a new function')

mc.remove_function(-1)
mc.print('After removing the last function')

mc.remove_dataset()
mc.print('After removing all training data')
repeat_function(f)
mc.print('After a few more function calls')

@mc.wrap_sensor()
def fcopy(x0, x1, x3):  # y
    return 3 * x0 + x1 + x3

repeat_function(fcopy, arity=3)
mc.print('After a few direct-sensor calls')

@mc.wrap_sensor('inverse')
def finv(y, x1, x2):  # x1
    return (y - x1 -  x2) / 3

repeat_function(finv, arity=3)
mc.print('After a few inverse-sensor calls')

globalx = 1
y = f(2, 3)
y.callback(100.0)
for kind in ('tdata', 'edata'):
    idx, out = mc.find_data([2, 3, 1], kind)
    if idx >= 0:
        print(f"Feedback callback: {y:.1f} updated to {out} in _{kind}['outputs'][{idx}] for inputs [2, 3, 1]")

repeat_function(mc, arity=3)
mc.print('After a few MC calls')

globalx = 1
y = f(2, 3)
mc.remove_dataset('tdata')
mc.remove_dataset('edata')
y.callback(100.0)

mc1 = ModelCaller(MCconfig(_ncargs=1))
mc1.print('A new unwrapped MC with one context argument', full=True)
mc1.add_model(mc.get_model(0), quality=True) # reuse model
repeat_function(mc1, arity=3)
mc1.print('After a few mc calls')

@mc_wrapd(auto_id=None)
def f2(x0, x1):  # y
    return 3 * x0 + x1
f2(10,11)
f2._mc.print('A new wrapped MC with only auto-id and no other context argument, after a function call', full=True)

m = LinearRegression()
m.fit([[1, 2, 3], [3, 4, 5]], [9, 10])
fpredict = mc_wrap(m.predict)  # wrapping a predefined function
fpredict([[10, 20, 30]])
fpredict._mc.print('A new MC, after wrapping a model.predict and calling MC', full=True)

m2 = LinearRegression()
m = mc_wrap(m, kind='model', auto_id=True)
mc2 = m._mc
mc2.merge_host()
mc2.train_all((np.array([[1, 2, 3], [3, 4, 5]]), np.array([9, 10])))
m(10, 20, 30)
mc2.print('A new MC, after wrapping a model and calling fit and predict', full=True)