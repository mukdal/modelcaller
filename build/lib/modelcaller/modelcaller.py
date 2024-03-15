"""modelcaller.py. Copyright (C) 2024, Mukesh Dalal <mukesh@aidaa.ai>"""
from dataclasses import dataclass
from inspect import currentframe
from random import random
import numpy as np
from sklearn.base import BaseEstimator
from torch import from_numpy
from torch.optim import AdamW
from torch.nn import Module
from torch.nn.functional import mse_loss
import logging
logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s')
logger = logging.getLogger(__name__)

class CallbackBase:
    """
    Facilitates adding callbacks to objects. It enables direct callback invocation and 
    delegates undefined attribute access to the encapsulated object.

    Attributes:
        value: The object enhanced with a callback.
        cbfn: The callback function to invoke.
    """
    def __init__(self, value, callback):
        """
        Initializes the CallbackBase with an object and its callback.

        Args:
            value: Object to which the callback is added.
            callback: Callback function.
        """
        self.value = value
        self.cbfn = callback

    def callback(self, *args, **kwargs):
        """
        Executes the callback function with given arguments.
        
        Returns:
            Result of the callback function.
        """
        return self.cbfn(*args, **kwargs)

    def __getattr__(self, item):
        """
        Delegates attribute access to the encapsulated object if not found in this class.
        
        Returns:
            The requested attribute from `value`.
        """
        return getattr(self.value, item)
   
    
class FloatCallback(CallbackBase, float):
    """
    Extends CallbackBase to support floats and integers with callbacks.
    It integrates callback functionality into float instances.
    """
    def __new__(cls, value, callback):
        """
        Creates a new FloatCallback instance, ensuring the value is a float or integer.

        Args:
            value (float|int): The float or integer value to enhance with a callback.
            callback (callable): The callback function to associate with this value.

        Returns:
            A new FloatCallback instance.

        Raises:
            AssertionError: If `value` is not a float or int.
        """
        assert isinstance(value, (float, int)), "non-float/int value passed to FloatCallback"
        obj = float.__new__(cls, value)  # Instantiate as float
        obj.cbfn = callback  # Attach callback function
        return obj
    
class StrCallback(CallbackBase, str):
    def __new__(cls, value, callback):
        assert isinstance(value, str), "non-str value passed to StrCallback"
        obj = str.__new__(cls, value)  # Instantiate as str
        obj.cbfn = callback  # Attach callback function
        return obj

class ArrayCallback(CallbackBase, np.ndarray):
    """
    Extends CallbackBase to support numpy arrays with callbacks.
    Enables callbacks on array operations or accesses.
    """
    def __new__(cls, input_array, callback):
        """
        Creates a new ArrayCallback instance from a numpy array.

        Args:
            input_array (np.ndarray): The numpy array to enhance with a callback.
            callback (callable): The callback function to associate with the array.

        Returns:
            A new ArrayCallback instance.
        """
        obj = np.asarray(input_array).view(cls)  # Create an instance viewed as ArrayCallback
        obj.cbfn = callback  # Attach callback function
        return obj

    def __array_finalize__(self, obj):
        """
        Finalizes the array, ensuring the callback is attached to new instances.

        This method is called automatically when new instances are created through slicing.

        Args:
            obj: The object from which to inherit the callback, if available.
        """
        if obj is None: return  # Exit if obj is None
        self.cbfn = getattr(obj, 'cbfn', None)  # Inherit or set default callback

@dataclass
class MCconfig: # default MC configuration
    """
    Default configuration for ModelCaller operations.

    Attributes:
        auto_cache (bool): Enable automatic caching of host call data.
        auto_id (bool): Automatically add MC id as the last context argument; None for thin MC with only auto_id.
        auto_mceval (bool): Automatically evaluate models after training.
        auto_mctrain (bool): Automatically train models after adding them.
        edata_fraction (float): Fraction of data cached for evaluation instead of training.
        feedback_fraction (float): Fraction of host calls randomly chosen for feedback.
        qlty_threshold (float): Quality (accuracy) threshold for models to be considered successful.
        _ncargs (int): Number of context arguments, excluding id.
    """
    auto_cache: bool = True
    auto_id: bool = False
    auto_mceval: bool = True
    auto_mctrain: bool = True
    edata_fraction: float = 0.3
    feedback_fraction: float = 0#0.1 
    qlty_threshold: float = 0.95
    _ncargs: int = 0

class ModelCaller():
    """
    Main ModelCaller class for managing model calls, including wrapping functions and models,
    and handling automatic training, evaluation, and data management.

    Attributes:
        Various configuration options (e.g., auto_cache, auto_mctrain) control the behavior of the MC.
    """
    def __init__(self, mcc=MCconfig()):
        """
        Initializes ModelCaller with a configuration object.

        Args:
            mcc (MCconfig):  ModelCaller Configuration.
        """
        super().__init__()
        self.auto_cache = False if mcc.auto_id == None else mcc.auto_cache
        self.auto_id = True if mcc.auto_id == None else mcc.auto_id
        self.auto_mceval = False if mcc.auto_id == None else mcc.auto_mceval
        self.auto_mctrain = False if mcc.auto_id == None else mcc.auto_mctrain
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
        self._tdata = {'inputs': np.array([], dtype=float), 'outputs': np.array([], dtype=float)}  # saved training data
        
    def __call__(self, *xs, wrapper=False, cvals=None):
        cvals = cvals or list()  # set default
        nensemble = (sum(self._qualities) + len(self._functions))
        assert nensemble > 0, "No models or functions in ModelCaller"
        if not wrapper: # called directly, not through a host  wrapper
            nfargs = len(xs) - self._ncargs # number of fn args
            xs, cvals = xs[:nfargs], xs[nfargs:] # split xs
        fsum = sum([f(*xs) for f in self._functions])
        msum = sum([model._mcpredict([list(xs)+list(cvals)]) for idx, model in enumerate(self._models) if self._qualities[idx]])
        res = (msum + fsum)[0] / nensemble
        if not wrapper: # called directly, not through a host wrapper
            res = self._process_result(res, xs, cvals)
        return res
    
    def __getattr__(self, item): # delegate any attribute not defined here to host
        return getattr(self._host, item)
       
    #def __str__(self): return self.fullstr(self, full=False)
   
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
        else:
            self._frame_model(model)
        self._models.append(model)
        self._qualities.append(quality)
        logger.info(f"After adding a model: {self.fullstr()}")
        idx = len(self._models) - 1
        if not quality and self.auto_mctrain:
            self.train_model(idx)
        return idx
   
    def eval_model(self, idx, all=False):
        if self._edata['inputs'].size == 0:
            return False
        model = self._models[idx]
        score = model._mceval(self._edata['inputs'], self._edata['outputs'])
        res = score >= self.qlty_threshold
        logger.info(f"Compare Model {idx} score = {score} with qlty_threshold {self.qlty_threshold} => quality = {res}")
        self._qualities[idx] = res
        if not all:  # not evaluating all models
            self._auto_call_target()
        return res
    
    def eval_all(self):
        if self._edata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.eval_model(idx, all=True)
            self._auto_call_target()

    def find_data(self, x, kind='tdata'):
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        idx = self._npindex(data['inputs'], x)
        y = data['outputs'][idx] if idx >= 0 else None
        return idx, y
      
    def fullstr(self, full=True, ntail=2): # string representation of MC with ntail inputs and outputs
        return f"{'auto_cache='+str(self.auto_cache) if full else ''}{', auto_id='+str(self.auto_id) if full else ''}{', auto_mceval='+str(self.auto_mceval) if full else ''}{', auto_mctrain='+str(self.auto_mctrain) if full else ''}{', edata_fraction='+str(self.edata_fraction) if full else ''}{', feedback_fraction='+str(self.feedback_fraction) if full else ''}{', qlty_threshold='+str(self.qlty_threshold) if full else ''}{', ncargs='+str(self._ncargs)+', ' if full else ''}call_target:{self._call_target}, #functions:{len(self._functions)}, model-qualities:{self._qualities}, #tdata:{len(self._tdata['inputs'])}, #edata:{len(self._edata['inputs'])}; tinputs:{'...' if len(self._tdata['inputs']) > ntail  else ''}{self._npp(self._tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self._tdata['outputs']) > ntail else ''}{self._npp(self._tdata['outputs'][-ntail:])}; einputs:{'...' if len(self._edata['inputs']) > ntail  else ''}{self._npp(self._edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self._edata['outputs']) > ntail else ''}{self._npp(self._edata['outputs'][-ntail:])}"
    
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
            logger.warning(f"Can't set call_target to {newstate} because no host")
        else:
            self._call_target = newstate 
 
    def train_model(self, idx, all=False):
        if self._tdata['inputs'].size > 0:
            model = self._models[idx]  
            self._models[idx]._mctrain(self._tdata['inputs'], self._tdata['outputs'])
            if not all and hasattr(self, 'auto_mceval') and self.auto_mceval:
                self.eval_model(idx)

    def train_all(self, dataset=None):
        if dataset != None:
            self.add_dataset(*dataset, kind='tdata')
        if self._tdata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.train_model(idx, all=True)
            if self.auto_mceval:
                self.eval_all()
     
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
                    y = host(*xs) if kind == 'function' else host._mcpredict([list(xs) + cvals]) # call fn or model
                    if self._call_target == 'both': 
                        y = (self(*xs, wrapper=True, cvals=cvals) + y) / 2 # call both MC and host
                y = self._process_result(y, xs, cvals=cvals)
                return y 
            self._host = host  # save original host function
            self._host_kind = kind 
            wrapper._mc = self  # save MC object in the wrapper
            if kind == 'model':
                self._frame_model(host)
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
                logger.warning(f"Feedback callback couldn't find the inputs {args} in {kind}")
        return inner
         
    @staticmethod
    def _frame_model(model):
        if isinstance(model, BaseEstimator):
            model._mcpredict = model.predict
            model._mctrain = model.fit
            model._mceval = model.score
        elif isinstance(model, Module):
            model._mcpredict = lambda x : torch_mcpredict(model, x)
            model._mctrain = lambda x,y : torch_mctrain(model, x, y)
            model._mceval = lambda x,y : torch_mceval(model, x, y)
    
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
            match y:
                case str(): y = StrCallback(y, self._callback(kind, *xs, *cvals))
                case np.ndarray(): y = ArrayCallback(y, self._callback(kind, *xs, *cvals))
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

def torch_mcpredict(model, x):
    model.train(False)
    xtype = type(x)
    if xtype == list: 
        x = np.array(x, dtype=np.float32)
    y = model(from_numpy(x)).detach().numpy()
    if xtype == list: 
        y = y.tolist()
    return y[0]  

def torch_mctrain(model, x, yt, epochs=100, lossfn=mse_loss, optimizer=AdamW):
    model.train()
    optimizer = optimizer(model.parameters())
    x = from_numpy(x)
    yt = from_numpy(yt)
    for i in range(epochs): 
        y = model(x)
        loss = lossfn(y, yt)
        optimizer.zero_grad()  # reset gradients
        loss.backward()
        optimizer.step()

def torch_mceval(model, x, yt):
    model.train(False)
    x = from_numpy(x)
    yt = from_numpy(yt)
    y = model(x)
    sum_error = (y - yt).pow(2).sum()
    yt_mean = yt.mean()
    sum_sqr = (yt - yt_mean).pow(2).sum()
    return (1 - sum_error / sum_sqr).item() 