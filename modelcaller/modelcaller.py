"""Copyright (C) 2024, Mukesh Dalal. All rights reserved. <mukesh@aidaa.ai>"""
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
    Facilitates adding a callback to a value. Don't use this class directly. Instead, use the subclass that's also the subclass of the value's desired class. The value will get encapsulated in the newly created object, which will mostly behave like the original value.
    """
    def __init__(self, value, callback):
        """
        value: Object to which the callback is added.
        callback: Callback function.
        """
        self.value = value
        self.cbfn = callback

    def callback(self, *args, **kwargs):
        """
        Executes the callback function with given arguments.
        """
        return self.cbfn(*args, **kwargs)

    def __getattr__(self, item):
        """
        Delegates attribute access to the encapsulated value if not found in this class.
        """
        return getattr(self.value, item)

    def __repr__(self):
        return str(type(self)) + ':(' + str(self.value) + ', ' + str(self.cbfn) + ')'   
    
class FloatCallback(CallbackBase, float):
    """
    Encapsulates a float or an int in a float object with a callback.
    """
    def __new__(cls, value, callback):
        assert isinstance(value, (float, int)), "non-float/int value passed to FloatCallback"
        obj = float.__new__(cls, value)  # Instantiate as float
        obj.cbfn = callback  # Attach callback function
        return obj
    
class StrCallback(CallbackBase, str):
    """
    Encapsulates a str in a str object with a callback.
    """
    def __new__(cls, value, callback):
        assert isinstance(value, str), "non-str value passed to StrCallback"
        obj = str.__new__(cls, value)  # Instantiate as str
        obj.cbfn = callback  # Attach callback function
        return obj

class ArrayCallback(CallbackBase, np.ndarray):
    """
    Encapsulates a ndarray in a ndarray object with a callback.
    """
    def __new__(cls, value, callback):
        assert isinstance(value, np.ndarray), "non-ndarray value passed to ArrayCallback"
        obj = np.asarray(value).view(cls)
        obj.cbfn = callback
        return obj

@dataclass
class MCconfig:
    """
    Default configuration for a ModelCaller object.
    auto_cache (bool): Enable automatic caching of host call data.
    auto_call_target (bool): Automatically change call target from 'host' to 'both' when a model qualifies after testing.
    auto_id (bool): Automatically add MC id as the last context argument; None for a thin MC with only auto_id capability.
    auto_eval (bool): Automatically evaluate models after their training.
    auto_train (bool): Automatically train models after adding them.
    edata_fraction (float): Fraction of data cached for evaluation instead of training.
    feedback_fraction (float): Fraction of host calls randomly chosen for supervisory feedback.
    qlty_threshold (float): Quality threshold for models to be considered qualified.
    _ncparams (int): Number of context parameters, excluding id.
    """
    auto_cache: bool = True
    auto_call_target: bool = True
    auto_id: bool = False
    auto_eval: bool = True
    auto_train: bool = True
    edata_fraction: float = 0.3
    feedback_fraction: float = 0#0.1 
    qlty_threshold: float = 0.95
    _ncparams: int = 0

class ModelCaller():  # abbreviated as MC
    """
    Facilitates calling, hosting, and registering models and functions with enhanced capabilities like automatic data sensing and caching, training, testing, and capturing supervisory and delayed feedback.
    """
    def __init__(self, mcc=MCconfig()):
        """
        mcc (MCconfig):  MC Configuration.
        """
        super().__init__()
        self.auto_cache = False if mcc.auto_id == None else mcc.auto_cache
        self.auto_call_target = False if mcc.auto_id == None else mcc.auto_call_target
        self.auto_id = True if mcc.auto_id == None else mcc.auto_id # subtlety!
        self.auto_eval = False if mcc.auto_id == None else mcc.auto_eval
        self.auto_train = False if mcc.auto_id == None else mcc.auto_train
        self.edata_fraction = mcc.edata_fraction 
        self.feedback_fraction = 0 if mcc.auto_id == None else mcc.feedback_fraction 
        self.qlty_threshold =  mcc.qlty_threshold 
        self._call_target = 'MC' # 'MC', 'host', or 'both': who to call when MC or the wrapped host is called?
        self._edata = {'inputs': np.array([]), 'outputs': np.array([])}  # saved evaluation data
        self._functions = [] # list of registered functions
        self._host = None   # original unwrapped host (gets populated by wrap_host method)
        self._host_kind = None # None, 'model', or 'function'
        self._models = [] # list of registered models
        self._ncparams = mcc._ncparams + (1 if self.auto_id else 0)
        self._qualified = [] # list of bools indicating whether corresponding model is qualified
        self._tdata = {'inputs': np.array([]), 'outputs': np.array([])}  # saved training data
        
    def __call__(self, *xs, wrapper=False, cargs=None):  # predict using MC
        """
        Calls registered models and functions and caches data based on configuration.
        xs (list): List of function arguments.
        wrapper (bool): True if called through a host wrapper.
        cargs (list): List of context arguments.
        """
        cargs = cargs or list()  # set default
        nensemble = (len(self._functions) + sum(self._qualified)) # number of functions and qualified models
        assert nensemble > 0, "No models or functions in ModelCaller"
        if not wrapper: # split xs into fn and context args
            nfargs = len(xs) - self._ncparams # number of fn args
            xs, cargs = xs[:nfargs], xs[nfargs:]
        fsum = sum([f(*xs) for f in self._functions])  # call functions
        msum = sum([model._mccall([list(xs)+list(cargs)]) for idx, model in enumerate(self._models) if self._qualified[idx]]) # call qualified models
        res = (msum + fsum)[0] / nensemble # average
        if not wrapper: # otherwise, this will be done in the host wrapper
            res = self._process_result(res, xs, cargs)
        return res
    
    def __getattr__(self, item):
        """
        Delegates undefined attribute access to the wrapped host, enabling seamless integration with host capabilities.
        """
        return getattr(self._host, item)
       
    def add_dataset(self, np_in, np_out, kind='tdata'): # training (default) or evaluation
        """
        Adds a dataset for training (default) or evaluation ('edata'), adding MC ID to inputs if configured.
        np_in (np.array): Inputs.
        np_out (np.array): Outputs.
        """
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        if self.auto_id:  # insert id into inputs
            id_array = np.full((np_in.shape[0], 1), id(self))
            np_in = np.concatenate((np_in, id_array), axis=1)
        data['inputs'] = np.concatenate((data['inputs'], np_in), axis=0) if data['inputs'].size else np_in
        data['outputs'] = np.concatenate((data['outputs'], np_out), axis=0) if data['outputs'].size else np_out
    
    def clear_dataset(self, kind='tdata'):
        """
        Clears all data from either the training or evaluation datasets.
        """
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        data['inputs'] = np.array([])
        data['outputs'] = np.array([])
    
    def eval(self):
        """
        Evaluates MC (not registered models) with available evaluation data.
        """
        if self._edata['inputs'].size == 0:
            return False
        outs = self(self._edata['inputs'])
        return self._r2_error(outs, self._edata['outputs'])
        
    def eval_model(self, idx, all=False):
        """
        Evaluates a registered model with available evaluation data, optionally updating the call_target.
        all (bool): True if called from eval_all_models
        """
        if self._edata['inputs'].size == 0:
            return False
        model = self._models[idx]
        score = model._mceval(self._edata['inputs'], self._edata['outputs'])
        res = score >= self.qlty_threshold
        logger.info(f"Compare Model {idx} score = {score} with qlty_threshold {self.qlty_threshold} => quality = {res}")
        self._qualified[idx] = res
        if not all and self.auto_call_target:  # not evaluating all models
            self._auto_call_target()
        return res
    
    def eval_all_models(self):
        """
        Evaluates all registered models, optionally updating the call_target.
        """
        if self._edata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.eval_model(idx, all=True)
            if self.auto_call_target:
                self._auto_call_target()

    def find_data(self, x, kind='tdata'):
        """
        Locates a specific input in the training or evaluation data, returning its index and output if found.
        """
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        idx = self._npindex(data['inputs'], x)
        y = data['outputs'][idx] if idx >= 0 else None
        return idx, y
      
    def fullstr(self, full=True, ntail=2):
        """
        Returns a string reptresentation of MC.
        full (bool): True for more attributes.
        ntail (int): number of last saved data points.
        """ 
        return f"{'auto_cache='+str(self.auto_cache) if full else ''}{', auto_id='+str(self.auto_id) if full else ''}{', auto_eval='+str(self.auto_eval) if full else ''}{', auto_train='+str(self.auto_train) if full else ''}{', edata_fraction='+str(self.edata_fraction) if full else ''}{', feedback_fraction='+str(self.feedback_fraction) if full else ''}{', qlty_threshold='+str(self.qlty_threshold) if full else ''}{', ncparams='+str(self._ncparams)+', ' if full else ''}call_target:{self._call_target}, #functions:{len(self._functions)}, model-qualities:{self._qualified}, #tdata:{len(self._tdata['inputs'])}, #edata:{len(self._edata['inputs'])}; tinputs:{'...' if len(self._tdata['inputs']) > ntail  else ''}{self._npp(self._tdata['inputs'][-ntail:])}; toutputs:{'...' if len(self._tdata['outputs']) > ntail else ''}{self._npp(self._tdata['outputs'][-ntail:])}; einputs:{'...' if len(self._edata['inputs']) > ntail  else ''}{self._npp(self._edata['inputs'][-ntail:])}; eoutputs:{'...' if len(self._edata['outputs']) > ntail else ''}{self._npp(self._edata['outputs'][-ntail:])}"
    
    def get_call_target(self):
        return self._call_target
        
    def get_host(self):
        return self._host
        
    def get_model(self, idx):
        return self._models[idx]
    
    def isqualified(self, idx):
        return self._qualified[idx]
        
    def register_function(self, fn=None):
        """
        Registers a function (default=host), allowing it to be called when MC is called. Returns index of the added function.
        """
        if fn == None: # add the original unwrapped host function
            assert self._host_kind == 'function', "host must be a function"
            fn = self._host
        self._functions.append(fn)
        return len(self._functions) - 1

    def register_host(self, qualified=True):
        """
        Registers the host and sets call_target so that host is called even when MC is called.
        """
        assert self._host != None, "no host to merge"
        idx = self.register_function() if self._host_kind == 'function' else self.register_model(qualified=qualified)
        self.update_call_target('MC')
        return idx, self._host_kind
        
    def register_model(self, model=None, qualified=False, model_api=None):
        """
        Registers a model (default=host), allowing it to be called when MC is called.
        Returns index of the added model and logs it.
        qualified (bool): True if the model is qualified.
        model_api (str): tuple (callfn, trainfn, evalfn) for model API.
        """
        if model == None: # add the original unwrapped host model
            assert self._host_kind == 'model', "host must be a model"
            model = self._host
        else:
            self._standardize(model, model_api)
        self._models.append(model)
        self._qualified.append(qualified)
        logger.info(f"After adding a model: {self.fullstr()}")
        idx = len(self._models) - 1
        if not qualified and self.auto_train:
            self.train_model(idx)
        return idx
       
    def remove_data(self, idx, kind='tdata'):
        """
        Removes a specific data item based on its index from training or evaluation data.
        """
        assert kind in ['tdata', 'edata'], "dataset kind must be 'tdata' or 'edata'"
        data = eval('self._'+ kind)
        data('self._'+ kind)['inputs'].pop(idx)
        data('self._'+ kind)['outputs'].pop(idx)

    def train(self, nested=True, dataset=None):
        """
        Trains MC, optionally adding a new dataset before training, in which case, optionally also train all registered models.
        """
        if dataset != None:
            self.add_dataset(*dataset, kind='tdata')
        # add local training code, if local parameters
        if nested:
            self.train_all_models()
    
    def train_model(self, idx, all=False):
        """
        Trains a registered model with available training data, optionally triggering an evaluation afterwards.
        all (bool): True if called from train_all_models
        """
        if self._tdata['inputs'].size > 0:
            model = self._models[idx]  
            self._models[idx]._mctrain(self._tdata['inputs'], self._tdata['outputs'])
            if not all and hasattr(self, 'auto_eval') and self.auto_eval:
                self.eval_model(idx)

    def train_all_models(self, dataset=None):
        """
        Trains all models managed by the ModelCaller, optionally adding a new dataset before training and triggering an evaluation afterwards.
        """
        if dataset != None:
            self.add_dataset(*dataset, kind='tdata')
        if self._tdata['inputs'].size > 0:
            for idx in range(len(self._models)):
                self.train_model(idx, all=True)
            if self.auto_eval:
                self.eval_all_models()
    
    def unregister_function(self, idx):
        """
        Removes a function from the registered list based on its index.
        """
        self._functions.pop(idx)
    
    def unregister_model(self, idx):
        """
        Removes a model from the registered and qualified lists based on its index.
        """
        self._models.pop(idx)
        self._qualified.pop(idx)
    
    def update_call_target(self, newstate):
        """
        Updates the call target, affecting how calls are directed between MC, host, or both.
        """
        assert newstate in ['host', 'both', 'MC'], "call_target must be 'host', 'both' or 'MC'"
        if self._host == None and newstate != 'MC':
            logger.warning(f"Can't set call_target to {newstate} because no host")
        else:
            self._call_target = newstate 
  
    def wrap_host(self, kind='function', cparams=None, model_api=None):
        """
        Wraps a host (model or function) with the ModelCaller, enhancing it with configured capabilities.
        cparams (list): list of context parameters
        model_api (str): tuple (callfn, trainfn, evalfn) for model API
        """
        cparams = cparams or list() # set default
        def decorator(host):
            def wrapper(*xs):
                self._ncparams = len(cparams)
                if  kind == 'function': # get context arguments from the caller frame
                    frame = currentframe().f_back
                    all_variables = {**frame.f_globals, **frame.f_locals}
                    cargs = [v for k, v in all_variables.items() if k in cparams]
                else: 
                    assert kind == 'model', "kind must be 'function' or 'model'"
                    nfargs = len(xs) - self._ncparams # number of fn args
                    xs, cargs = xs[:nfargs], list(xs[nfargs:]) # split xs
                if self.auto_id:
                    self._ncparams += 1
                    cargs.append(id(self))
                if self._call_target == 'MC':
                    y = self(*xs, wrapper=True, cargs=cargs) # call only MC
                else:
                    y = host(*xs) if kind == 'function' else host._mccall([list(xs) + cargs]) # call fn or model
                    if self._call_target == 'both': 
                        y = (self(*xs, wrapper=True, cargs=cargs) + y) / 2 # call both MC and host
                y = self._process_result(y, xs, cargs=cargs)
                return y 
            self._host = host  # save original host function
            self._host_kind = kind 
            wrapper._mc = self  # save MC object in the wrapper
            if kind == 'model':
                self._standardize(host, model_api)
            return wrapper
        self._call_target = 'host'  # initial call target
        return decorator   
   
    def wrap_sensor(self, skind='direct'):
        """
        Wraps a sensor around the decorated function g, enabling additional data capture.
        """
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
     
    def _add_data(self, out, *ins): # add ins=>out to training or eval data based on edata_fraction
        kind = 'edata' if random() <= self.edata_fraction else 'tdata'
        data = eval('self._'+ kind)
        ins_np = np.array(ins).reshape(1, -1)  # Reshape xs to a 2D array with one row
        out_np = np.array([out])  # Make y a 1D array with a single element
        if data['inputs'].size == 0:
            data['inputs'] = ins_np
            data['outputs'] = out_np
        else:
            data['inputs'] = np.concatenate((data['inputs'], ins_np), axis=0)
            data['outputs'] = np.concatenate((data['outputs'], out_np), axis=0)
        return kind
    
    def _around(self, xl, places=1): # recursively round floats
        if type(xl) in (list, tuple): 
            return [self._around(x, places) for x in xl]
        if isinstance(xl, np.ndarray):
            if np.issubdtype(xl.dtype, np.floating):
                return np.around(xl, places)
        if isinstance(xl, float):
            return round(xl, places)
        return xl
    
    def _auto_call_target(self): # automatically update call target
        if (len(self._functions) + sum(self._qualified)) == 0: # no functions or qualified models
            self._call_target = 'host'
        elif self._call_target == 'host' and self.auto_call_target: 
            self._call_target = 'both'
                         
    def _callback(self, kind, *args): # callback function for feedback (currently only output override)
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

    def _get_feedback(self, y, *xs, cargs=None): # get supervisory feedback
        cargs = cargs or list()  # set default
        if random() <= self.feedback_fraction:
            newy = input(f"x={self._around(xs)}, context={self._around(cargs)}, {self._around(y)}. To override y, type a new valueand return, otherwise just press return:")
            if newy != '':  
                return type(y)(newy)
        return y  
   
    @staticmethod
    def _npindex(data, *key): # find row index of first match of key in data, else -1
        if data.size == 0: 
            return -1
        npkey = np.array(key)
        matches = np.all(data == npkey, axis=1)
        idxs = np.where(matches)[0]
        if idxs.size > 0:
            return idxs[0]
        return -1
     
    def _npp(self, arr): # round and convert np array to string
        return '[' + ', '.join([str(row.tolist()) for row in self._around(arr)]) + ']'
    
    def _process_result(self, out, ins, cargs=None): # enhance ins=> out with supervisory override, caching, and feedback callback 
        cargs = cargs or list()  # set default
        out = self._get_feedback(out, *ins, cargs=cargs)
        if self.auto_cache:
            kind = self._add_data(out, *ins, *cargs)
            match out:
                case str(): out = StrCallback(out, self._callback(kind, *ins, *cargs))
                case np.ndarray(): out = ArrayCallback(out, self._callback(kind, *ins, *cargs))
                case float() | int(): out = FloatCallback(out, self._callback(kind, *ins, *cargs))
                case _: logger.error(f"output {out} of {type(out)} not supported by CallbackBase")
        return out    
         
    def _standardize(self, model, model_api): # add standard model interface (_mc* methods)
        if model_api != None:
            model._mccall = model_api[0]
            model._mctrain = model_api[1]
            model._mceval = model_api[2]
        elif isinstance(model, BaseEstimator): # sklearn model
            model._mccall = model.predict
            model._mctrain = model.partial_fit if hasattr(model, 'partial_fit') else model.fit
            model._mceval = model.score
        elif isinstance(model, Module): # pytorch model
            model._mccall = lambda x : self._torch_mccall(model, x)
            model._mctrain = lambda x,y : self._torch_mctrain(model, x, y)
            model._mceval = lambda x,y : self._torch_mceval(model, x, y)
        elif isinstance(model, ModelCaller): # nested
            model._mccall = model
            model._mctrain = model.train(data='supervised')
            model._mceval = model.eval(data='supervised')
        else: 
            logger.error(f"ModelCaller: {type(model)} models require model_api argument of the form (call_function, train_function, eval_function)")

    @staticmethod
    def _torch_mccall(model, x):
        """Predict using a PyTorch model"""
        model.train(False)
        xtype = type(x)
        if xtype == list: 
            x = np.array(x, dtype=np.float32)
        y = model(from_numpy(x)).detach().numpy()
        if xtype == list: 
            y = y.tolist()
        return y[0]  

    @staticmethod
    def _torch_mctrain(model, x, yt, epochs=100, lossfn=mse_loss, optimizer=AdamW):
        """Train a PyTorch model"""
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

    def _torch_mceval(self, model, x, yt):
        """Evaluate a PyTorch model, returning a R2 score"""
        model.train(False)
        x = from_numpy(x)
        yt = from_numpy(yt)
        y = model(x)
        return self._r2_error(y, yt)
    
    @staticmethod
    def _r2_error(self, y, yt):
        """return R2 score between y and yt"""
        sum_error = (y - yt).pow(2).sum()
        yt_mean = yt.mean()
        sum_sqr = (yt - yt_mean).pow(2).sum()
        return (1 - sum_error / sum_sqr).item() 
    
def wrap_mc(host, kind='function', cparams=None, **config):
    """Wrap a predefined model or function as a host in a new ModelCaller object with optional context parameters and configuration."""
    return ModelCaller(MCconfig(**config)).wrap_host(kind, cparams)(host)

def decorate_mc(kind='function', cparams=None, **config):
    """Decorate a model or function definition for wrapping it as a host in a new ModelCaller object with optional context parameters and configuration."""
    def wrapper(host):
        return wrap_mc(host, kind, cparams, **config)  
    return wrapper