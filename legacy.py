"""ModelBox"""

from random import random
import os
import math
import time
import queue
from threading import Thread, Event
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler as Scaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_selection import f_classif, SelectKBest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import openai
from inspect import signature

production = False  # set True if production, False otherwise (faster)
torch.manual_seed(0)  # uncomment for reproducibility

# from config file (YAML)
combiner_default = 'error_weighted_mean'
models_default = [[nn.Linear, torch.optim.AdamW]]
lossfn_default = torch.nn.MSELoss
control_default = 'parallel'
senseQ_default = None
generateQ_default = None

class ModelBox(nn.Module):  #v1
    def __init__(self, models, ins, outs, combiner=torch.mean, lossfn=torch.nn.MSELoss()):
        super().__init__()
        self.models = models  # list of [model, optimizer] pairs
        self.ins = ins # input shape for each model
        self.outs = outs # output shape for each model
        self.tdata = None    # training data
        self.vdata = None    # validation data
    def forward(self, x): pass
    def train(self, x, y): pass
    def validate(self, x, y): pass
    def addmodel(self, model, optimizer): pass
    def removemodel(self, model): pass

class ModelBox(nn.Module):  # v2
    def __init__(self, senseQ=senseQ_default, generateQ=generateQ_default, models=models_default, control=control_default, combiner=combiner_default, lossfn=lossfn_default): # [role: all]
        super().__init__()
        self.models = models  # list of full models (including all relevant info)
        self.tdata = None    # training data
        self.edata = None    # evaluation data
    def forward(self, x): pass # [role: tech] predict/generate y from x; x may be batched
    def feedback(self, x, yt): pass # [role: tech] provide <x, yt> feedback on some previous forward(x)
    def train(self, x, yt): pass # [role: tech] train on <x, yt> pair 
    def evaluate(self, x, yt): pass # [role: tech] evaluate on <x, yt> pair
    def async_loop(self): pass  # [role: tech/power] get input from senseQ and put output on generateQ
    def add_model(self, model): pass # [role: tech/power] add, train & evalute new model
    def remove_model(self, model): pass # [role: tech/power] remove model from self.models  

class Sys():
    """ ATIM (sub)system with optional subsystems, training function, etc. """
    def __init__(self):
        self.name = name
        self.fn = fn                # (prediction) function with exactly one output (use __call__ instead of direct call)
        self.subs = subs            # list of top-level subsystems
        self.tfn = tfn              # training function
        self.gfn = None             # generative internal version of self.fn (use self.gen() for external calls)
        self.twins = twins          # neural twins
        self.type = type            # 'pytorch', 'numpy'
        self.vtype = vtype          # output value type: real, binary, multiclass  (add other vtypes as needed)
        self.data = None            # ground truth data for training twins
        self.pdata = None           # predicted data for testing trainable twins
        self.ins = None             # width of training input (auto set when data or pdata first saved)
        self.outs = None            # width of training output (auto set when data or pdata first saved)
        self.state = 0              # 0:prep, 1:record, 2:train, 3:integrated
        self.itime = time.time_ns() # init time

    def __call__(self, *args):  # each arg may have a different width
        y = self.fn(*args)
        self._save(y, *args)
        return y
    
    def gen(self, *args):
        """ generative integrated alternative to __call__ """
        class _Thread(Thread):
            threads = []    # list of threads created by this function
            def __init__(self, sys, afn, q, e, isgen, *xs):
                Thread.__init__(self)
                self.threads.append(self)
                self.sys = sys
                self.afn = afn
                self.q = q
                self.isgen = isgen
                self.xs = xs       
                e.set()
            def _put(self, y):
                if istnsr(y):
                    self.q.put((y.penalty(), random(), y)) # random needed for forcing strict ordering
                else:
                    self.q.put((0, random(), y))
            def run(self):
                if self.isgen:
                    for y in self.afn(*self.xs):
                        #print(f"{self.sys.name}: gen out {torch.reshape(y,(-1,))[:3]}... of shape {list(y.shape)}")
                        y = aggerr(y, self.xs, vtype=self.sys.vtype)  # afn is not a twin
                        self._put(y)
                else:
                    y = self.afn(*self.xs)
                    #print(f"{self.sys.name}: out {torch.reshape(y,(-1,))[:3]}... of shape {list(y.shape)}")
                    if not isinstance(self.afn,Twin):  # afn is not a twin
                        y = aggerr(y, self.xs, vtype=self.sys.vtype)
                    self._put(y)

        args = initerr(tuple([mdtensor(x) for x in args]))
        if istnsr(args):
            arg = aggerr(torch.cat(args, dim=1), args)  # adding torch.cat to Tnsr autoprop does not seem to work (??)
            #print(f"{self.name}: in {torch.reshape(arg,(-1,))[:3]}... of shape {list(arg.shape)}")
        q = queue.PriorityQueue()
        e = Event()
        for twin in self.twins: # run twins before numpy detach (does this matter?)
            if twin.state > 0:
                #print(f"{self.name}: twin in {torch.reshape(arg,(-1,))[:3]}... of shape {list(arg.shape)}")
                _Thread(self, twin, q, e, False, arg).start()
        if self.gfn != None:
            #print(f"{self.name}: gfn in {torch.reshape(args[0],(-1,))[:3]}... of shape {list(args[0].shape)}")
            _Thread(self, self.gfn, q, e, True, *args).start()
        else:
            if self.type == 'numpy': 
                args = tuple([a.detach() for a in args])
            #print(f"{self.name}: fn in {torch.reshape(args[0],(-1,))[:3]}... of shape {list(arg[0].shape)}")
            _Thread(self, self.fn, q, e, False, *args).start()
        e.wait()
        cpenalty = Tnsr.errmax   # current penalty initialized to max
        while any([t.is_alive() for t in _Thread.threads]) or not q.empty():
            penalty, _, y = q.get()
            y = mdtensor(y)
            #print(f"{self.name}: yield {torch.reshape(y,(-1,))[:3]}... of shape {list(y.shape)}")
            if penalty == 0:   # no error
                self._save(y, *args)
            if not production or penalty < cpenalty:
                cpenalty = penalty
                yield y

    def train(self, *args): # the last arg is the target output
        if self.trainable():
            yp = self.tfn(*args)
            #print(f"{self.name}: {yp}")
            if self.state > 0:  # save data and pdata
                self.data = self.mdata(self.data, args[-1], *args[:-1])
                self.pdata = self.mdata(self.pdata, yp, *args[:-1])
            return yp  

    def changestate(self, state, gfn=None):
        if state == 3 and gfn != None:
            self.gfn = gfn
        self.state = state        
        for sub in self.subs:
            sub.changestate(state)
    
    def createtwins(self, mdict):  # requires self.ins and self.outs to be correctly set
        if self.name in mdict and (torch.is_tensor(self.data) or torch.is_tensor(self.pdata)):
            for model in mdict[self.name]:
                twin = Twin(self, model)
                if self.twins == []:    # direct append does not seem to work (??)
                    self.twins = [twin]
                else: self.twins += [twin]
            #print(f"{self.name}: twins={[t.name for t in self.twins]}")
        for sub in self.subs:
            sub.createtwins(mdict)

    def traintwins(self):
        for twin in self.twins:
            twin.train()
        for sub in self.subs:
            sub.traintwins()

    def testtwins(self):
        for twin in self.twins:
            twin.test()
        for sub in self.subs:
            sub.testtwins()  
    
    def trainable(self):
        return self.tfn != None
    
    def mdata(self, data, y, *args): # merge <time, *args, y> rows to self.data for training
        x = torch.cat(tuple([mdtensor(x) for x in args]),dim=1)
        y = mdtensor(y)
        ng = torch.cat((torch.full((x.size(dim=0),1),time.time_ns()-self.itime),x,y),dim=1)
        ng = ng.detach()    # don't use in any backprop
        if torch.is_tensor(data):
            data = torch.cat((data,ng),dim=0)
        else:
            data = ng
            self.ins = x.size(dim=1)
            self.outs = y.size(dim=1)
        #print(f"{self.name}: saving data {torch.reshape(data,(-1,))[:3]}... of shape {list(ng.shape)}")
        return data   

    def delete(self):
        for sub in self.subs:
            sub.delete()
        for twin in self.twins:
            del twin
        self.twins = []
        self.gfn = None
        self.data = None
        self.pdata = None
        del self

    def __repr__(self):
        str = f"{self.name}: (state={self.state}, ins={self.ins}, outs={self.outs}, type={self.type}, vtype={self.vtype}, #data={rows(self.data)}, #pdata={rows(self.pdata)}, subs={[s.name for s in self.subs]}, twins={[t for t in self.twins]})"
        for sub in self.subs:
            str += "\n" + sub.__repr__()
        return str    
        
    def _save(self, y, *args):
        if self.state > 0 and isarray(y) and isarray(args):    # save data or pdata
            if self.trainable():
                self.pdata = self.mdata(self.pdata, y, *args)
            else:
                self.data = self.mdata(self.data, y, *args)




        



class Tnsr(torch.Tensor):
    """ Adds errors and probabilities to tensors and propagate them semi-automatically """

    # class vars
    errmax = torch.finfo(torch.float32).max     # max possible error (sqrt(loss))
    autoprop = {torch.round, torch.as_tensor, torch.reshape, torch.Tensor.detach, torch.Tensor.__getitem__}    # automatically propagate vtype, err, lp, lperr (write custom code for others)

    @staticmethod
    def __new__(cls, x, vtype='real', lp=0, err=0, lperr=0, *args, **kwargs):  # does not seem to work without this (??)
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, vtype='real', lp=0, err=0, lperr=0):
        super().__init__()
        self.vtype = vtype    # value type: real, binary, etc.
        self.lp = lp          # log probability of being generated as output
        self.err = err        # error in value
        self.lperr = lperr    # log probability error

    def __repr__(self):
        return f"{super().__repr__()}, vtype={self.vtype}, penalty={self.penalty()}, lp={self.lp}, err={self.err}, lperr={self.lperr})"
    
    def penalty(self):
        p = math.exp(self.lp)
        return p*self.err + (1-p) * self.errmax
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        vtypes = [x.vtype for x in list(args)+list(kwargs.values()) if hasattr(x,'vtype')]
        vtype = vtypes[0] if vtypes != [] else 'real'  # assumes other vtypes to be identical
        err = sum([x.err for x in list(args)+list(kwargs.values()) if hasattr(x,'err')],0)
        lp = sum([x.lp for x in list(args)+list(kwargs.values()) if hasattr(x,'lp')],0)
        lperr = sum([x.lperr for x in list(args)+list(kwargs.values()) if hasattr(x,'lperr')],0)
        ret = super().__torch_function__(func, types, args, kwargs)
        if func in cls.autoprop:
            ret = Tnsr(ret, vtype=vtype, lp=lp, err=err, lperr=lperr)
        return ret



class Data(Dataset):
    """ Data set for batch loading in twin training """
    def __init__(self, x, y):
        with torch.no_grad():
            self.data = x
            self.labels = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def isarray(x):  # is x a numpy array or a torch tensor or a recursive list or tuple of them?
    if isinstance(x, list) or isinstance(x, tuple): 
        return(all([isarray(y) for y in x]))
    else:
        return isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)
  
def istnsr(x):  # is x a Tnsr or a (recursive) list or tuple of all Tnsrs?
    if isinstance(x, list) or isinstance(x, tuple): 
        return(all([istnsr(y) for y in x]))
    else:
        return isinstance(x, Tnsr)

def initerr(x, vtype='real', lp=0, err=0, lperr=0):   # recursively create tnsrs, for those not already tnsrs
    if isinstance(x, list): 
        return([initerr(y, vtype, lp, err, lperr) for y in x])
    if isinstance(x, tuple):
        return(tuple([initerr(y, vtype, lp, err, lperr) for y in x]))
    else:
        if isarray(x) and not hasattr(x, 'err'):
            x = mdtensor(x)
            return Tnsr(x, vtype, lp, err, lperr)
        else:
            return x
    
def aggerr(y, xs, vtype='real'):  # ensure that y is a tnsr that aggregates xs' errs, lps, lperrs
    y = mdtensor(y)
    if isarray(y) and not hasattr(y, 'err'): 
        y = Tnsr(y, vtype=vtype)
        y.err = sum([x.err for x in xs if hasattr(x,'err')],0)
        y.lp = sum([x.lp for x in xs if hasattr(x,'lp')],0)
        y.lperr = sum([x.lperr for x in xs if hasattr(x,'lperr')],0)
    return y

def mdtensor(y):    # convert y to a rightly formatted tensor, if not already so
    if isarray(y):
        y = torch.as_tensor(y, dtype=torch.float32) # if not tensor, convert
        if y.dim() == 1:    # if 1D, convert to 2D
            y = torch.reshape(y,(y.size(dim=0),-1)) 
    return y

def rows(x):    # number of rows in a tensor
    if torch.is_tensor(x):
        return x.size(dim=0)
    return 0

def linmodel():  # default linear neural model
    return torch.nn.Linear

def nonlinmodel(sh=20):   # default non-linear neural model
    def model(si=1, so=1):  
        return torch.nn.Sequential(
                    torch.nn.Linear(si, sh),
                    torch.nn.ReLU(),
                    torch.nn.Linear(sh, so))
    return model

def bcmodel(sh=10): # default binary classification neural model
    def model(si=1, so=1):  
        return torch.nn.Sequential(
                    torch.nn.Linear(si, sh),
                    torch.nn.ReLU(),
                    torch.nn.Linear(sh, so),
                    torch.nn.Sigmoid())
    return model

def ccfddemo():  # credit card fraud detection (trainable) demo
    def normalizer(X, features: list):
        scaler = Scaler()
        X.loc[:, features] = scaler.fit_transform(X.loc[:, features])
        return X

    def data_loader(path='data/creditcard.csv'):
        #print("Loading data...")
        df = pd.read_csv(path,dtype=np.float32)
        fraud = df[df['Class'] == 1] #separates the fraud instances
        non_fraud = df[df['Class'] == 0].sample(n=2*fraud.shape[0]) #samples 984 non-fruad instances
        df = pd.concat([fraud, non_fraud]).sample(frac=1) #concatinates these two and shuffles the rows
        X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
        xt, xv, yt, yv = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
        xt = normalizer(xt, ['Time', 'Amount'])
        xv = normalizer(xv, ['Time', 'Amount'])
        return xt.values, xv.values, yt.values, yv.values    

    def fit_predict(f):
        def local(*args):
            return f.fit(*args).predict(*args[:-1])
        return local
    
    def gccfd(x):
        for s in select.gen(x):
            for c in classify.gen(s):
                yield c

    mdict = {   'ccfd': (bcmodel(),), 
                'select': (linmodel(),), 
                'classify': (bcmodel(),)
    }

    print("\n* CCFD demo:")    
    select = SelectKBest(f_classif, k=20)
    classify = AdaBoostClassifier()
    ccfd = Pipeline([('select', select),('classify', classify)])  # does not use subsystems (different than a2e)!

    print(f"\nDeconstruct & reconstuct the base system:")    
    select = Sys('select', select.transform, tfn=select.fit_transform, type='numpy')
    classify = Sys('classify', classify.predict, tfn=fit_predict(classify), type='numpy', vtype='binary')
    ccfd = Sys('ccfd', ccfd.predict, [select, classify], fit_predict(ccfd), type='numpy', vtype='binary')
    print(ccfd)

    print(f"\nTrain the base system:")
    ccfd.changestate(1)
    xt, xv, yt, yv = data_loader()   # training and validation data sets
    print(f"Input: numpy {xt.reshape(-1)[:3]}... of shape {np.shape(xt)}")
    yp = ccfd.train(xt, yt)
    mt = select(xt)  # needed since ccfd does not call select
    print(f"Features: numpy {mt.reshape(-1)[:3]}... of shape {np.shape(mt)}")
    print(f"Output: numpy {yt.reshape(-1)[:3]}... of shape {np.shape(yt)}")
    classify.pdata = classify.mdata(classify.mdata, yp, mt)
    print(ccfd)

    print(f"\nCreate twins:")
    ccfd.changestate(2)
    ccfd.createtwins(mdict)
    print(ccfd)

    print(f"\nTrain the twins using saved data:")
    ccfd.traintwins()
    print(ccfd)

    print(f"\nTest the base system and the twins:")
    print(f"Input: numpy {xv.reshape(-1)[:3]}... of shape {np.shape(xv)}")
    yp = ccfd(xv)
    mv = select(xv)
    print(f"Features: numpy {mv.reshape(-1)[:3]}... of shape {np.shape(mv)}")
    print(f"Output: numpy {yp.reshape(-1)[:3]}... of shape {np.shape(yp)}")
    classify.pdata = classify.mdata(classify.pdata, yp, mv)
    print(f'base Accuracy: {accuracy_score(yv, yp):.4f}, Recall: {recall_score(yv, yp):.4f}, Precision: {precision_score(yv, yp):.4f}, F1: {f1_score(yv, yp):.4f}')
    ccfd.testtwins() 
    print(ccfd)  

    print(f"\nActivate production twins:")
    ccfd.changestate(3, gccfd)
    print(f"Input: numpy {xv.reshape(-1)[:3]}... of shape {np.shape(xv)}")
    mv = select(xv)
    for yp in ccfd.gen(xv):
        print(f"Output {torch.reshape(yp,(-1,))[:3]}... of shape {list(yp.shape)}")
        classify.pdata = classify.mdata(classify.pdata, yp, mv)
        yp = yp.detach()
        #yp = np.rint(yp)
        print(f'... accuracy: {accuracy_score(yv, yp):.4f}, Recall: {recall_score(yv, yp):.4f}, Precision: {precision_score(yv, yp):.4f}, F1: {f1_score(yv, yp):.4f}')
    print(ccfd)  

def a2xdemos(n):   # a2x demos (synthetic regression)
    def sb(x): 
        time.sleep(1)
        return 3*x+4

    def sc(x): 
        time.sleep(1)
        return 2*x-5

    def sd(x): 
        #time.sleep(1)
        return x+7

    def se(x, y): 
        time.sleep(1)
        return 2*x+y   

    mdict = {   'sa': (nonlinmodel(20),), 
            'sb': (linmodel(),), 
            'sc': (linmodel(),), 
            'sd': (linmodel(),), 
            'se': (linmodel(),),
            'sm': (linmodel(),)
    }    

    def a2xbody(sa, ga, mdict=mdict, n=n, sm=None, gm=None, *args):
        print(f"\nDeconstruct the base system:")
        print(sa)

        print(f"\nReconstruct and run the base system:")
        sa.changestate(1)
        xt = Tnsr(torch.rand(n, 1, requires_grad=False))
        print(f"Input {torch.reshape(xt,(-1,))[:3]}... of shape {list(xt.shape)}")
        yt = sa(xt, *args)
        if isarray(yt):
            print(f"Output {torch.reshape(yt,(-1,))[:3]}... of shape {list(yt.shape)}")
        print(sa)

        print(f"\nCreate twins:")
        sa.changestate(2)
        sa.createtwins(mdict)
        print(sa)

        print(f"\nTrain the twins using saved data:")
        sa.traintwins()
        print(sa)

        print(f"\nTest the twins:")
        xv = Tnsr(torch.rand(n, 1, requires_grad=False))
        print(f"Input {torch.reshape(xv,(-1,))[:3]}... of shape {list(xv.shape)}")
        yv = sa(xv, *args)
        if isarray(yv):
            print(f"Output {torch.reshape(yv,(-1,))[:3]}... of shape {list(yv.shape)}")
        sa.testtwins()
        print(sa)

        print(f"\nActivate production twins:")
        if sm != None:
            sm.gfn = gm
        sa.changestate(3, ga)
        xt = Tnsr(torch.rand(n, 1, requires_grad=False))
        print(f"Input {torch.reshape(xt,(-1,))[:3]}... of shape {list(xt.shape)}")
        for yt in sa.gen(xt, *args):
            if isarray(yt):
                print(f"Output {torch.reshape(yt,(-1,))[:3]}... of shape {list(yt.shape)}")
        print(sa)

    def a2edemo(sb=sb, sc=sc, sd=sd, se=se):    # 4 subsystems with a split-and-join data flow
        def sa(a):
            b = sb(a)
            c = sc(b)
            d = sd(b)
            e = se(c, d)
            return e
        
        sb, sc, sd, se = tuple([Sys(sx.__name__, sx) for sx in (sb, sc, sd, se)])
        sa = Sys('sa', sa, [sb,sc,sd,se])

        def ga(a):
            for b in sb.gen(a):
                for c,d in zip(sc.gen(b), sd.gen(b)):
                    for e in se.gen(c,d):
                        yield e
    
        print("\n* A2E demo:")
        a2xbody(sa, ga)    
    
    def a2cdemo(sb=sb,sc=sc):   # 2 subsystems with a linear data flow 
        def sa(a):
            b = sb(a)
            c = sc(b)
            return c
        
        sb, sc = tuple([Sys(sx.__name__, sx) for sx in (sb, sc)])
        sa = Sys('sa', sa, [sb,sc])

        def ga(a):
            for b in sb.gen(a):
                for c in sc.gen(b):
                    yield c

        print("\n* A2C demo:")
        a2xbody(sa, ga)

        print("\n* A2C2 demo:")      # multiple twins of a substsystem
        sa.delete()
        
        mdict['sa'] += (linmodel(),)
        a2xbody(sa, ga)    

    def a2bdemo(sb=sb):   # multilevel nesting of subsystems
        def sm(a):
            b = sb(a)
            return b
        
        def sa(a):
            m = sm(a)
            return m
               
        sb = Sys('sb', sb)
        sm = Sys('sm', sm, [sb])
        sa = Sys('sa', sa, [sm])

        def gm(a):
            for b in sb.gen(a):
                yield b
            
        def ga(a):
            for m in sm.gen(a):
                yield m

        print("\n* A2B demo:")
        a2xbody(sa, ga, sm=sm, gm=gm) 

    def genaidemo(): # generative ai demo
        def whisper(afile):
            afile = open(afile, "rb")
            text = openai.Audio.transcribe("whisper-1", afile)['text']
            print(f"whisper: output={text}")
            return text
        
        def chatgpt(prompt):
            r = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                              {"role": "user", "content": prompt}
                        ])['choices'][0]['message']['content']
            print(f"chatgpt: output={r}")
            return r

        def dalle(prompt):
            print(f"dalle: input={prompt}")
            r = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="256x256")['data'][0]['url']
            print(f"dalle: output={r}")
            return r
        
        def sb(x): 
            return 3*x+4
        
        def sa(x, a):
            p = whisper(a)
            q = chatgpt(p)
            y = sb(x)
            r = q + " and number " + str(round(y[0,0].item()))
            s = dalle(r)
            return s
        
        sb = Sys('sb', sb)
        sa = Sys('sa', sa, [sb])

        def ga(x, a):
            p = whisper(a)
            q = chatgpt(p)
            for y in sb.gen(x):
                r = q + " and number " + str(round(y[0,0].item()))
                s = dalle(r)
            yield s

        mdict = {'sb': (linmodel(),)}
        print("\n* Generative AI demo:")
        a2xbody(sa, ga, mdict, n, None, None, "data/command.mp3")     

    a2bdemo()
    #a2edemo()
    #a2cdemo()   # this changes mdict
    #genaidemo()

def fdemo(fa = lambda x1, x2: x1 / x2, n=1000, m=100): # synthetic regression of a function f trained on n data points
    nargs = len(signature(fa).parameters)
    sa = Sys('sa', fa)
    def ga(*args):
        for y in sa(*args):
            yield y  
    mdict = {'sa': (nonlinmodel(20),),}
  
    print(f"\nDeconstruct the base system:")
    print(sa)

    print(f"\nReconstruct and run the base system:")
    sa.changestate(1)
    xt = tuple([Tnsr(torch.rand(n, 1, requires_grad=False)*m) for _ in range(nargs)])    
    for x in xt: print(f"Input {torch.reshape(x,(-1,))[:3]}... of shape {list(x.shape)}")
    yt = sa(*xt)
    if isarray(yt):
        print(f"Output {torch.reshape(yt,(-1,))[:3]}... of shape {list(yt.shape)}")
    print(sa)

    print(f"\nCreate twins:")
    sa.changestate(2)
    sa.createtwins(mdict)
    print(sa)

    print(f"\nTrain the twins using saved data:")
    sa.traintwins()
    print(sa)

    print(f"\nTest the twins:")
    xv = tuple([Tnsr(torch.rand(n, 1, requires_grad=False)*m) for _ in range(nargs)])
    for x in xv: print(f"Input {torch.reshape(x,(-1,))[:3]}... of shape {list(x.shape)}")
    yv = sa(*xv)
    if isarray(yv):
        print(f"Output {torch.reshape(yv,(-1,))[:3]}... of shape {list(yv.shape)}")
    sa.testtwins()
    print(sa)

    """  fix error here, before uncommenting
    print(f"\nActivate production twins:")
    sa.changestate(3, ga)
    xt = tuple([Tnsr(torch.rand(n, 1, requires_grad=False)*m) for _ in range(nargs)])    
    for x in xt: print(f"Input {torch.reshape(x,(-1,))[:3]}... of shape {list(x.shape)}")
    for yt in sa.gen(*xt):
        if isarray(yt):
            print(f"Output {torch.reshape(yt,(-1,))[:3]}... of shape {list(yt.shape)}")
    print(sa)
    """

if __name__ =='__main__':
    #fdemo()
    #ccfddemo()
    a2xdemos(1024)