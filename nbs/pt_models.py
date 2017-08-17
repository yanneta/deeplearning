from imports import *
from torch_imports import *

def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))

def V(x): return Variable(x.cuda(async=True))
def VV(x): return Variable(x.cuda(async=True), volatile=True)
def to_np(v): return v.data.cpu().numpy()
def pred_batch(m, x): return to_np(m(VV(x)))

def SGD_Momentum(momentum): 
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def set_learning_rate(opt, lr):
    for group in opt.param_groups: group['lr'] = lr

def trainable_params(m):
    return [p for p in m.parameters() if p.requires_grad]

def set_trainable_attr(m,b): m.trainable=b
    
def set_trainable(l, b):
    l.apply(lambda m: set_trainable_attr(m,b))
    for p in l.parameters(): p.requires_grad = b

def cond_init(m, init_fn):
    if isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    else:
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias'): m.bias.data.fill_(1.)

def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)
    
Flatten = Lambda(lambda x: x.view(x.size(0), -1))

def cut_model(m, cut): return list(m.children())[:cut]

def predict_to_bcolz(m, gen, arr, workers=4):
    lock=threading.Lock()
    m.eval()
    for x,*_ in tqdm(gen):
        arr.append(pred_batch(m, x))
        with lock: arr.flush()

def num_features(m): 
    if hasattr(m, 'num_features'): return m.num_features
    elif hasattr(m, 'out_features'): return m.out_features
    else: return num_features(children(m)[-1])

def accuracy(preds, targs):
    preds = np.argmax(preds, axis=1)
    return (preds==targs).mean()

def accuracy_thresh(thresh): 
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def get_probabilities(net, loader):
    net.eval()
    return np.vstack(pred_batch(net, data) for data, *_ in loader)

def step(m, opt, x, y, crit):
    loss = crit(m(V(x)), V(y))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.data[0]

def set_train(m):
    if len(children(m))>0: return
    if hasattr(m, 'running_mean') and not m.trainable: m.eval()
    else: m.train()
        
def fit(m, data, epochs, crit, opt, metrics=[], callbacks=[]):
    avg_mom=0.95
    
    for epoch in trange(epochs, desc='Epoch'):
        avg_loss=None
        m.apply(set_train)
        #m.eval()
        t = tqdm(data.trn_dl)
        for x,y in t:
            loss = step(m,opt,x,y, crit)
            if avg_loss is None: avg_loss = loss
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            t.set_postfix(loss=avg_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(avg_loss)
            if stop: return
            
        m.eval()
        res=np.zeros((len(metrics),), dtype=np.float32)
        loss=0.
        for x,y in data.val_dl:
            preds,targs = m(VV(x)),VV(y)
            res += np.array([f(to_np(preds),to_np(targs)) for f in metrics])
            loss += crit(preds, targs)
        res /= len(data.val_dl)
        loss /= len(data.val_dl)
        res = [avg_loss, to_np(loss)[0]] + list(res)
        print(res)
        stop=False
        for cb in callbacks: stop = stop or cb.on_epoch_end(res)
        if stop: return

def predict(m, dl, res=[]):
    m.eval()
    for x,y in dl: res.append(pred_batch(m, x))
    return res
