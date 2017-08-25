from imports import *
from torch_imports import *
from layer_optimizer import *

def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))

def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias'): m.bias.data.fill_(0.)

def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

Flatten = Lambda(lambda x: x.view(x.size(0), -1))

def cut_model(m, cut): return list(m.children())[:cut]

def predict_to_bcolz(m, gen, arr, workers=4):
    lock=threading.Lock()
    m.eval()
    for x,*_ in tqdm(gen):
        y = to_np(m(VV(x)).data)
        with lock:
            arr.append(y)
            arr.flush()

def num_features(m): 
    c=children(m)
    if hasattr(c[-1], 'num_features'): return c[-1].num_features
    elif hasattr(c[-1], 'out_features'): return c[-1].out_features
    if hasattr(c[-2], 'num_features'): return c[-2].num_features
    elif hasattr(c[-2], 'out_features'): return c[-2].out_features
    return num_features(children(m)[-1])

def accuracy(preds, targs):
    preds = np.argmax(preds, axis=1)
    return (preds==targs).mean()

def accuracy_thresh(thresh): 
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9):
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)
    precision = tp / (y_pred.sum(dim=1)+eps)
    recall = tp / (y_true.sum(dim=1)+eps)
    return torch.mean(
        precision*recall / (precision*(beta**2)+recall+eps) * (1+beta**2))

def get_probabilities(net, loader):
    net.eval()
    return np.vstack(net(VV(data)) for data, *_ in loader)

def step(m, opt, xs, y, crit):
    loss = crit(m(*V(xs)), V(y))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.data[0]

def set_train_mode(m):
    if hasattr(m, 'running_mean') and not (hasattr(m,'trainable') and m.trainable): m.eval()
    else: m.train()

def fit(m, data, epochs, crit, opt, metrics=None, callbacks=None):
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    
    for epoch in trange(epochs, desc='Epoch'):
        avg_loss=None
        apply_leaf(m, set_train_mode)
        t = tqdm(data.trn_dl)
        for (*x,y) in t:
            loss = step(m,opt,x,y, crit)
            if avg_loss is None: avg_loss = loss
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            t.set_postfix(loss=avg_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(avg_loss)
            if stop: return
            
        vals = validate(m, data.val_dl, crit, metrics)
        print(np.round([avg_loss] + vals, 6))
        stop=False
        for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
        if stop: return

def validate(m, dl, crit, metrics):
    preds,targs = predict_with_targs(m, dl)
    loss=crit(preds,targs).data[0]
    preds,targs = to_np(preds),to_np(targs)
    res = [f(preds,targs) for f in metrics]
    return [loss] + res

def predict(m, dl):
    m.eval()
    return torch.cat([m(*VV(x)) for *x,_ in dl]).data.cpu()

def predict_with_targs(m, dl):
    m.eval()
    preda,targa = zip(*[(m(*VV(x)),y) for *x,y in dl])
    return torch.cat(preda).data.cpu(), torch.cat(targa)
