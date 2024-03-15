#!/usr/bin/env python
# coding: utf-8

# Copyright (C) 2024, Mukesh Dalal. 
# <mukesh@aidaa.ai>

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import random
import numpy as np
from modelcaller import ModelCaller, MCconfig, mc_wrapd, mc_wrap
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
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


# In[ ]:


@mc_wrapd(cargs=['globalx'])
def f(x0, x1): 
    global globalx
    return 3 * x0 + x1 + globalx


# In[ ]:


mc = f._mc
print('A new wrapped MC with one context argument:', mc.fullstr())


# In[ ]:


repeat_function(f)
print('After a few function calls:', mc.fullstr(full=False))


# In[ ]:


from sklearn.linear_model import LinearRegression
mc.add_model(LinearRegression())
print('After training and evaluating the added model: ', mc.fullstr(full=False))


# In[ ]:


repeat_function(f)
print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


if mc.get_call_target() == 'both': 
    mc.merge_host()
    print('After merging host function: ', mc.fullstr(full=False))
    repeat_function(f)
    print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


from sklearn.neural_network import MLPRegressor
midx = mc.add_model(MLPRegressor(hidden_layer_sizes=(), activation='identity'))
print('After training and evaluating the added model: ', mc.fullstr(full=False))
repeat_function(f)
print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


if mc.get_call_target() == 'MC':
    xy = generate_data(mc.get_host())
    mc.add_dataset(xy[0], xy[1])
    print('After adding more data but before training: ', mc.fullstr(full=False))
    mc.train_all()
    print('After training and evaluating with the new data: ', mc.fullstr(full=False))
    repeat_function(f)
    print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


if mc.get_model_quality(midx) == False:
    mc.qlty_threshold = -100
    print('After updating qlty_threshold: ', mc.fullstr())
    mc.eval_all()
    print('After reevaluating all models with the new threshold: ', mc.fullstr(full=False))
    repeat_function(f)
    print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


mc.remove_model(1)
mc.qlty_threshold = 0.95
print('After removing the second model and reverting the threshold: ', mc.fullstr())


# In[ ]:


fidx = mc.add_function(lambda x: x * x)
print('After adding a new function: ', mc.fullstr(full=False))


# In[ ]:


mc.remove_function(fidx)
print('After removing the last function: ', mc.fullstr(full=False))


# In[ ]:


mc.remove_dataset()
print('After removing all training data: ', mc.fullstr(full=False))
repeat_function(f)
print('After a few more function calls: ', mc.fullstr(full=False))


# In[ ]:


@mc.wrap_sensor()
def fcopy(x0, x1, x3):  # y
    return 3 * x0 + x1 + x3


# In[ ]:


repeat_function(fcopy, arity=3)
print('After a few direct-sensor calls: ', mc.fullstr(full=False))


# In[ ]:


@mc.wrap_sensor('inverse')
def finv(y, x1, x2):  # x1
    return (y - x1 -  x2) / 3

repeat_function(finv, arity=3)
print('After a few inverse-sensor calls: ', mc.fullstr(full=False))


# In[ ]:


globalx = 1
y = f(2, 3)
y.callback(100.0)
for kind in ('tdata', 'edata'):
    idx, out = mc.find_data([2, 3, 1], kind)
    if idx >= 0:
        print(f"Feedback callback: {y:.1f} updated to {out} in _{kind}['outputs'][{idx}] for inputs [2, 3, 1]")


# In[ ]:


repeat_function(mc, arity=3)
print('After a few MC calls: ', mc.fullstr(full=False))


# In[ ]:


globalx = 1
y = f(2, 3)
mc.remove_dataset('tdata')
mc.remove_dataset('edata')
y.callback(100.0)


# In[ ]:


mc1 = ModelCaller(MCconfig(_ncargs=1))
print('A new unwrapped MC with one context argument: ', mc1.fullstr())
mc1.add_model(mc.get_model(0), quality=True) # reuse model
repeat_function(mc1, arity=3)
print('After a few mc calls: ', mc1.fullstr(full=False))


# In[ ]:


import torch
import torch.nn as nn
mc1.add_model(nn.Linear(3,1), quality=True)
print('After adding a pytorch model: ', mc1.fullstr(full=False))
repeat_function(mc1, arity=3)
print('After a few mc calls: ', mc1.fullstr(full=False))


# In[ ]:


@mc_wrapd(auto_id=None)
def f2(x0, x1):  # y
    return 3 * x0 + x1
f2(10,11)
print('A new wrapped MC with only auto-id and no other context argument, after a function call: ', f2._mc.fullstr())


# In[ ]:


m = LinearRegression()
m.fit([[1, 2, 3], [3, 4, 5]], [9, 10])
fpredict = mc_wrap(m.predict)  # wrapping a predefined function
fpredict([[10, 20, 30]])
print('A new MC, after wrapping a model.predict and calling MC: ', fpredict._mc.fullstr())


# In[ ]:


m2 = LinearRegression()
m = mc_wrap(m, kind='model', auto_id=True)
mc2 = m._mc
mc2.merge_host()
mc2.train_all((np.array([[1, 2, 3], [3, 4, 5]], dtype=float), np.array([9, 10], dtype=float)))
m(10, 20, 30)
print('A new MC, after wrapping a model and calling fit and predict: ', mc2.fullstr())


# In[ ]:


import os
import requests
HF_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@mc_wrapd()
def llm(prompt):
    response = requests.post(API_URL, headers=headers, json=prompt)
    return response.json()[0]['generated_text']

llm("I want to")
llm("I do not want to")
print(llm._mc.fullstr())

