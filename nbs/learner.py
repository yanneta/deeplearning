from imports import *
from torch_imports import *
from fast_gen import *
from pt_models import *
from dataset_pt import *
from sgdr_pt import *
from layer_optimizer import *

def split_parms(m, split_at):
    c=children(m)
    c0=children(c[0])
    return (c0[:split_at], c0[split_at:], c[1:])

class ConvnetBuilder():
    model_meta = {
        resnet18:[8,5], resnet34:[8,5], resnet50:[8,5], resnet101:[8,5], resnext50:[8,5],
        wrn:[8,5], dn121:[10,6], inceptionresnet_2:[5,9], inception_4:[19,8]
    }

    def __init__(self, f, c, is_multi, ps=None, xtra_fc=None, xtra_cut=0):
        self.f,self.c,self.is_multi,self.xtra_cut = f,c,is_multi,xtra_cut
        self.ps = ps or [0.25,0.5]
        self.xtra_fc = xtra_fc or [512]
        
        cut,self.lr_cut = self.model_meta[self.f]
        cut-=xtra_cut
        layers = cut_model(self.f(True), cut)
        self.nf=num_features(layers[-1])*2
        layers += [AdaptiveConcatPool2d(), Flatten]
        self.top_model = nn.Sequential(*layers).cuda()

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.ps, list): self.ps = [self.ps]*n_fc
            
        fc_layers = self.get_fc_layers()
        self.fc_model = nn.Sequential(*fc_layers).cuda()
        apply_init(self.fc_model, kaiming_normal)
        self.model=nn.Sequential(*([self.top_model]+fc_layers)).cuda()

    @property
    def name(self): return f'{self.f.__name__}_{self.xtra_cut}'
    
    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn())
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU)
            ni=nf
        final_actn = nn.Sigmoid if self.is_multi else nn.LogSoftmax
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def set_mom(self, m): 
        if isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)): m.momentum=0.01
            
    def get_layer_groups(self, do_fc=False):
        if do_fc: m,lr_cut = self.fc_model,0
        else:     m,lr_cut = self.model,   self.lr_cut
        return m, split_parms(m,lr_cut)


class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None):
        self.data,self.models,self.metrics_ = data,models,metrics
        self.sched=None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = os.path.join(self.data.path, tmp_name)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.models_path = os.path.join(self.data.path, models_name)
        if not os.path.exists(self.models_path): os.mkdir(self.models_path)
        self.crit = F.binary_cross_entropy if data.is_multi else F.nll_loss
        
        self.save_fc1()
        self.freeze()

    @classmethod
    def pretrained_convnet(self, f, data, ps=None, xtra_fc=None, xtra_cut=0, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut)
        return Learner(data, models, **kwargs)
                           
    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)

    def num_features(self): return num_features(self.models.model)
    
    def set_data(self, data):
        self.data = data
        self.save_fc1()
        self.freeze()
    
    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val')]
        if os.path.exists(names[0]) and not force: 
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf,n) for n in names]
        
    def save_fc1(self):
        self.get_activations()
        act, val_act = self.activations
        
        if len(self.activations[0])==0:
            m=self.models.top_model
            predict_to_bcolz(m, self.data.fix_dl, act)
            predict_to_bcolz(m, self.data.val_dl, val_act)
            
        self.fc_data = ImageClassifierData.from_arrays(self.data.path, 
               (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs, classes=self.data.classes)

    def __getitem__(self,i): return self.children[i]
        
    @property
    def children(self): return children(children(self.models.model)[0])
        
    def freeze_to(self, n):
        c=self.children
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def unfreeze(self): self.freeze_to(0)
    def freeze(self): self.freeze_to(9999)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    def save(self, name): save_model(self.models.model, self.get_model_path(name))
    def load(self, name): load_model(self.models.model, self.get_model_path(name))
    
    def train(self, epochs): self.data.train(self.models.model, epochs)
    def train_fc(self, epochs): self.fc_data.train(self.models.fc_model, epochs)
        
    @property
    def metrics(self):
        if self.metrics_ is None:
            return [accuracy_multi] if self.data.is_multi else [accuracy]
        return self.metrics_
    
    def get_cycle_end(self, name):
        if name is None: return None
        return lambda sched, cycle: self.save_cycle(name, cycle)
    
    def save_cycle(self, name, cycle): self.save(f'{name}_cyc_{cycle}')
    def load_cycle(self, name, cycle): self.load(f'{name}_cyc_{cycle}')
        
    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None,
                metrics=None, callbacks=None):
        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics
        if cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl)*cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched: self.sched=LossRecorder(layer_opt)
        callbacks+=[self.sched]
        for cb in callbacks: cb.on_train_begin()
        n_epoch = sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle)
        fit(model, data, n_epoch, self.crit, layer_opt.opt, metrics, callbacks)

    def get_layer_opt(self, do_fc, lrs, wds):
        m,layer_groups = self.models.get_layer_groups(do_fc)
        return m,LayerOptimizer(self.opt_fn, layer_groups, lrs, wds)

    def fit(self, lrs, n_cycle, do_fc=False, wds=None, **kwargs):
        self.sched = None
        data = self.fc_data if do_fc else self.data
        m,layer_opt = self.get_layer_opt(do_fc, lrs, wds)
        self.fit_gen(m, data, layer_opt, n_cycle, **kwargs)
        
    def lr_find(self, do_fc=False, start_lr=1e-5, end_lr=10, wds=None):
        self.save('tmp')
        m,layer_opt = self.get_layer_opt(do_fc, start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr)
        data = self.fc_data if do_fc else self.data
        self.fit_gen(m, data, layer_opt, 1)
        self.load('tmp')

    def predict(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        return to_np(predict(self.models.model, dl).data)

    def TTA(self, n_aug=4, is_test=False):
        m = self.models.model
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(m, dl1)
        preds1 = [to_np(preds1)]*math.ceil(n_aug/4)
        preds2 = [to_np(predict(m, dl2)) for i in range(n_aug)]
        return np.stack(preds1+preds2).mean(0), to_np(targs)
