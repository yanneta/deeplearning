from imports import *
from torch_imports import *

def sum_geom(a,r,n):
    return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

conv_dict = {np.dtype('int32'): torch.IntTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}

def T(a):
    a = np.array(a)
    return conv_dict[a.dtype](a)

def V_(x):  return x.cuda(async=True) if isinstance(x, Variable) else Variable(x.cuda(async=True))
def V(x):   return [V_(o) for o in x] if isinstance(x,list) else V_(x)
def VV_(x): return x.cuda(async=True) if isinstance(x, Variable) else Variable(x.cuda(async=True), volatile=True)
def VV(x):  return [VV_(o) for o in x] if isinstance(x,list) else VV_(x)

def to_np(v):
    if isinstance(v, Variable): v=v.data
    return v.cpu().numpy()

def children(m): return list(m.children())

def split_by_idxs(seq, idxs):
    last, sl = 0, len(seq)
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def SGD_Momentum(momentum):
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]

def chain_params(p): return list(chain(*[trainable_params_(o) for o in p]))

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)
    else: f(m)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))


def opt_params(parm, lr, wd):
    return {'params': chain_params(parm), 'lr':lr, 'weight_decay':wd}

class LayerOptimizer():
    def __init__(self, opt_fn, layer_groups, lrs, wds=None):
        if not isinstance(layer_groups[0], Iterable): layer_groups=[layer_groups]
        if not isinstance(lrs, Iterable): lrs=[lrs]
        if len(lrs)==1: lrs=lrs*len(layer_groups)
        if wds is None: wds=0.
        if not isinstance(wds, Iterable): wds=[wds]
        if len(wds)==1: wds=wds*len(layer_groups)
        self.layer_groups,self.lrs,self.wds = layer_groups,lrs,wds
        self.opt = opt_fn(self.opt_params())

    def opt_params(self):
        params = list(zip(self.layer_groups,self.lrs,self.wds))
        return [opt_params(*p) for p in params]

    @property
    def lr(self): return self.lrs[-1]

    def set_lrs(self, lrs):
        self.lrs=lrs
        set_lrs(self.opt, lrs)

def set_lrs(opt, lrs):
    if not isinstance(lrs, Iterable): lrs=[lrs]
    if len(lrs)==1: lrs=lrs*len(opt.param_groups)
    for pg,lr in zip(opt.param_groups,lrs): pg['lr'] = lr
