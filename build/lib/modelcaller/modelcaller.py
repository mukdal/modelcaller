"""mc.py. Copyright (C) 2024, Mukesh Dalal <mukesh@aidaa.ai>"""

import numpy as np
from inspect import currentframe
from random import random
from dataclasses import dataclass

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
        obj.cbfn = callback  # Assign the callback directly to this float instance
        return obj
    
class ArrayCallback(CallbackBase, np.ndarray): # for arrays
    def __new__(cls, input_array, callback):
        obj = np.asarray(input_array).view(cls)
        obj.cbfn = callback
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.cbfn = getattr(obj, 'cbfn', None)

@dataclass
class MCconfig: # default MC configuration
    auto_cache: bool = True # auto cache host call data?
    auto_id: bool = False # automatically add MC id as the last context arg; if None then just thin MC with only auto_id
    auto_test: bool = True # auto test after training a model?
    auto_train: bool = True # auto train after adding a model?
    edata_fraction: float = 0.3 # fraction of data cached for evaluation (instead of training)
    feedback_fraction: float = 0 #0.1 # fraction of host calls randomly chosen for feedback    
    qlty_threshold: float = 0.95 # quality (accuracy) threshold for models
    _ncargs: int = 0 # number of context args, except id (model args = function args + context args)

class ModelCaller(): # main MC class
    def __init__(self, mcc=MCconfig()):
        super().__init__()
        self.auto_cache = False if mcc.auto_id == None else mcc.auto_cache
        self.auto_id = True if mcc.auto_id == None else mcc.auto_id
        self.auto_test = False if mcc.auto_id == None else mcc.auto_test
        self.auto_train = False if mcc.auto_id == None else mcc.auto_train
        self.edata_fraction = mcc.edata_fraction 
        self.feedback_fraction = 0 if mcc.auto_id == None else mcc.feedback_fraction 
        self.qlty_threshold =  mcc.qlty_threshold 
        self._call_target = 'MC' # 'MC', 'host', or 'both': who to call when MC or the wrapped host is called?
        self._edata = {'inputs': np.array([]), 'outputs': np.array([])}  # saved evaluation data
        self._functions = [] # list of functions (same number of args)
        self._host = None   # original unwrapped host (gets populated by mc_wrap)
        self._host_kind = None # None, 'model', or 'function'
        self._models = [] # list of models (same number of args)
        self._ncargs = mcc._ncargs + (1 if self.auto_id else 0)
        self._qualities = [] # list of model qualities
        self._tdata = {'inputs': np.array([]), 'outputs': np.array([])}  # saved training data
        
    def __call__(self, *xs, wrapper=False, cvals=None):
        cvals = cvals or list()  # set default
        if not wrapper: # called directly, not through a host  wrapper
            nfargs = len(xs) - self._ncargs # number of fn args
            xs, cvals = xs[:nfargs], xs[nfargs:] # split xs
        fsum = sum([f(*xs) for f in self._functions])
        msum = sum([model.predict([list(xs)+list(cvals)]) for idx, model in enumerate(self._models) if self._qualities[idx]])
        res = (msum + fsum)[0] / (sum(self._qualities) + len(self._functions))
        if not wrapper: # called directly, not through a host wrapper
            res = self._process_result(res, xs, cvals)
        return res
    
    def __getattr__(self, item): # delegate any attribute not defined here to host
        return getattr(self._host, item)
    
    def add_dataset(self, npx, npy, kind='tdata'): # training (default) or evaluation
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        if self.auto_id:  # insert id into inputs
            id_array = np.full((npx.shape[0], 1), id(self))
            npx = np.concatenate((npx, id_array), axis=1)
        data['inputs'] = np.concatenate((data['inputs'], npx), axis=0) if data['inputs'].size else npx
        data['outputs'] = np.concatenate((data['outputs'], npy), axis=0) if data['outputs'].size else npy
    
    def add_function(self, fn=None):
        if fn == None: # add the original unwrapped host function
            fn = self._host
        self._functions.append(fn)
        return len(self._functions) - 1 # return index of the added function
    
    def add_model(self, model=None, quality=False):
        if model == None:
            model = self._host
        self._models.append(model)
        self._qualities.append(quality)
        self.print('After adding a model')
        idx = len(self._models) - 1
        if not quality and self.auto_train:
            self.train_model(idx)
        return idx

    def find_data(self, x, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        idx = self._npindex(data['inputs'], x)
        y = data['outputs'][idx] if idx >= 0 else None
        return idx, y
    
    def get_call_target(self):
        return self._call_target
        
    def get_host(self):
        return self._host
        
    def get_model(self, idx):
        return self._models[idx]
    
    def get_model_quality(self, idx):
        return self._qualities[idx]
    
    def merge_host(self):
        assert self._host != None, "no host to merge"
        idx = self.add_function() if self._host_kind == 'function' else self.add_model(quality=True)
        self.set_call_target('MC')
        return idx, self._host_kind
    
    def print(self, tag, full=False, ntail=2): # print tag string and then MC with ntail inputs and outputs
        print(f"{tag}: {'auto_cache='+str(self.auto_cache) if full else ''}{', auto_id='+str(self.auto_id) if full else ''}{', auto_test='+str(self.auto_test) if full else ''}{', auto_train='+str(self.auto_train) if full else ''}{', edata_fraction='+str(self.edata_fraction) if full else ''}{', feedback_fraction='+str(self.feedback_fraction) if full else ''}{', qlty_threshold='+str(self.qlty_threshold) if full else ''}{', ncargs='+str(self._ncargs)+', ' if full else ''}call_target:{self._call_target}, #functions:{len(self._functions)}, model-qualities:{self._qualities}, #tdata:{len(self._tdata['inputs'])}, #edata:{len(self._edata['inputs'])}; tinputs:{'...' if len(self._tdata['inputs']) > ntail  else ''}{self._npp(self._tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self._tdata['outputs']) > ntail else ''}{self._npp(self._tdata['outputs'][-ntail:])}; einputs:{'...' if len(self._edata['inputs']) > ntail  else ''}{self._npp(self._edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self._edata['outputs']) > ntail else ''}{self._npp(self._edata['outputs'][-ntail:])}")
    
    def remove_data(self, idx, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        data('self._'+ kind)['inputs'].pop(idx)
        data('self._'+ kind)['outputs'].pop(idx)
        
    def remove_dataset(self, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        data['inputs'] = np.array([])
        data['outputs'] = np.array([])
    
    def remove_function(self, idx):
        self._functions.pop(idx)
    
    def remove_model(self, idx):
        self._models.pop(idx)
        self._qualities.pop(idx)
    
    def set_call_target(self, newstate):
        assert newstate in ['host', 'both', 'MC'], "call_target must be 'host', 'both' or 'MC'"
        if self._host == None and newstate != 'MC':
            print(f"Warning: can't set call_target to {newstate} because no host")
        else:
            self._call_target = newstate 
    
    def test_model(self, idx, all=False):
        if self._edata['inputs'].size == 0:
            return False
        model = self._models[idx]
        score = model.score(self._edata['inputs'], self._edata['outputs'])
        res = score >= self.qlty_threshold
        print(f"Compare Model {idx} score = {score} with qlty_threshold {self.qlty_threshold} => quality = {res}")
        self._qualities[idx] = res
        if not all:  # not testing all models
            self._auto_call_target()
        return res
    
    def test_all(self):
        if self._edata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.test_model(idx, all=True)
            self._auto_call_target()

    def train_model(self, idx, all=False):
        if self._tdata['inputs'].size > 0:  
            self._models[idx].fit(self._tdata['inputs'], self._tdata['outputs'])
            if not all and hasattr(self, 'auto_test') and self.auto_test:
                self.test_model(idx)

    def train_all(self, dataset=None):
        if dataset != None:
            self.add_dataset(*dataset, kind='tdata')
        if self._tdata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.train_model(idx, all=True)
            if self.auto_test:
                self.test_all()
     
    def wrap_host(self, kind='function', cargs=None): # wrap this MC around a host (model or function) with optional context  (replacing the current host)
        cargs = cargs or list() # set default
        def decorator(host):
            def wrapper(*xs):
                self._ncargs = len(cargs)
                if  kind == 'function': # get context values from the caller frame
                    frame = currentframe().f_back
                    all_variables = {**frame.f_globals, **frame.f_locals}
                    cvals = [v for k, v in all_variables.items() if k in cargs]
                else: 
                    assert kind == 'model', "kind must be 'function' or 'model'"
                    nfargs = len(xs) - self._ncargs # number of fn args
                    xs, cvals = xs[:nfargs], list(xs[nfargs:]) # split xs
                if self.auto_id:
                    self._ncargs += 1
                    cvals.append(id(self))
                if self._call_target == 'MC':
                    y = self(*xs, wrapper=True, cvals=cvals) # call only MC
                else:
                    y = host(*xs) if kind == 'function' else host.predict([list(xs) + cvals]) # call fn or model
                    if self._call_target == 'both': 
                        y = (self(*xs, wrapper=True, cvals=cvals) + y) / 2 # call both MC and host
                y = self._process_result(y, xs, cvals=cvals)
                return y 
            self._host = host  # save original host function
            self._host_kind = kind 
            wrapper._mc = self  # save MC object in the wrapper
            return wrapper
        self._call_target = 'host'  # initial call target
        return decorator   
   
    def wrap_sensor(self, skind='direct'): # wrap a sensor around some function g
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
     
    def _add_data(self, y, *xs):
        kind = 'edata' if random() <= self.edata_fraction else 'tdata'
        data = eval('self._'+ kind)
        xs_np = np.array(xs).reshape(1, -1)  # Reshape xs to a 2D array with one row
        y_np = np.array([y])  # Make y a 1D array with a single element
        if data['inputs'].size == 0:
            data['inputs'] = xs_np
            data['outputs'] = y_np
        else:
            data['inputs'] = np.concatenate((data['inputs'], xs_np), axis=0)
            data['outputs'] = np.concatenate((data['outputs'], y_np), axis=0)
        return kind
    
    def _around(self, xl, places=1):
        if type(xl) in (list, tuple): 
            return [self._around(x, places) for x in xl]
        return np.around(xl, places) 
    
    def _auto_call_target(self):
        if sum(self._qualities) == 0: 
            self._call_target = 'host'
        elif self._call_target == 'host': 
            self._call_target = 'both'
                         
    def _callback(self, kind, *args):
        def inner(y):
            nonlocal self, kind, args
            data = eval('self._'+ kind)
            idx = self._npindex(data['inputs'], *args)
            if idx >= 0:
                data['outputs'][idx] = y
                return y
            else:
                print("Warning: feedback callback couldn't find the inputs", args, "in", kind)
        return inner
         
    def _get_feedback(self, y, *xs, cvals=None):
        cvals = cvals or list()  # set default
        if random() <= self.feedback_fraction:
            newy = input(f"x={self._around(list(xs))}, context={self._around(list(cvals))}, {y=:.1f}. To override y, type a new value (float) and return, otherwise just press return:")
            if newy != '':  
                return float(newy)
        return y  
   
    @staticmethod
    def _npindex(data, *key): # find row index of first match, else -1
        if data.size == 0: 
            return -1
        npkey = np.array(key)
        matches = np.all(data == npkey, axis=1)
        idxs = np.where(matches)[0]
        if idxs.size > 0:
            return idxs[0]
        return -1
     
    def _npp(self, arr):
        return '[' + ', '.join([str(row.tolist()) for row in self._around(arr)]) + ']'
    
    def _process_result(self, y, xs, cvals=None):   
        cvals = cvals or list()  # set default
        y = self._get_feedback(y, *xs, cvals=cvals)
        if self.auto_cache:
            kind = self._add_data(y, *xs, *cvals)
            match type(y):
                case np.ndarray: y = ArrayCallback(y, self._callback(kind, *xs, *cvals))
                case _: y = FloatCallback(y, self._callback(kind, *xs, *cvals))
        return y    

def mc_wrap(host, kind='function', cargs=None, **config):
    """Wrap a (predefined) model or function in a new ModelCaller object with optional context and config."""
    return ModelCaller(MCconfig(**config)).wrap_host(kind, cargs)(host)

def mc_wrapd(kind='function', cargs=None, **config):
    """Decorator for wrapping a model or function (being defined) in a new ModelCaller object with optional context and config."""
    def wrapper(host):
        return mc_wrap(host, kind, cargs, **config)  
    return wrapper