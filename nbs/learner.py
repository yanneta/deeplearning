from imports import *
from torch_imports import *
from fast_gen import *
from pt_models import *
from dataset_pt import *
from sgdr_pt import *
        
class Learner():
    f_cuts = {
        resnet18:8, resnet34:8, resnet50:8, resnet101:8, resnext50:8,
        wrn:8, dn121:10, inceptionresnet_2:5, inception_4:19
    }

    def __init__(self, f, data, opt_fn=optim.RMSprop, tmp_name='tmp', models_name='models', 
                 metrics=None, ps=None, xtra_fc=[]):
        self.f,self.data,self.opt_fn,self.metrics_,self.ps,self.xtra_fc = f,data,opt_fn,metrics,ps,xtra_fc
        self.tmp_path = os.path.join(self.data.path, tmp_name)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.models_path = os.path.join(self.data.path, models_name)
        if not os.path.exists(self.models_path): os.mkdir(self.models_path)
        self.crit = F.binary_cross_entropy if data.is_multi else F.cross_entropy
        self.create()

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), 
                            chunklen=1, mode='w', rootdir=name)

    def num_features(self): return num_features(self.model)
    
    def get_activations(self, sz, force=False):
        tmpl = f'_{self.f.__name__}_{self.data.sz}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val')]
        if os.path.exists(names[0]) and not force: 
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(sz,n) for n in names]
        
    def save_fc1(self, sz):
        self.get_activations(sz)
        if len(self.activations[0]): return
        predict_to_bcolz(self.top_model, self.data.fix_dl, self.activations[0])
        predict_to_bcolz(self.top_model, self.data.val_dl, self.activations[1])

    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn())
        return res

    def get_fc_layers(self, nf):
        res=[]
        ni=nf
        for nf in self.xtra_fc:
            res += self.create_fc_layer(ni, nf, p=self.ps[0], actn=nn.ReLU)
            ni=nf
        final_actn = nn.Sigmoid if self.data.is_multi else nn.Logsoftmax
        res += self.create_fc_layer(ni, self.data.c, p=self.ps[0], actn=final_actn)
        return res

    def create(self):
        cut = self.f_cuts[self.f]
        layers = cut_model(self.f(True), cut)
        nf=num_features(layers[-1])
        layers += [nn.AdaptiveAvgPool2d((1,1)), Flatten]
        self.top_model = nn.Sequential(*layers).cuda()
        
        if not self.ps: self.ps=[0]
        elif not isinstance(self.ps, list): self.ps = [self.ps]
            
        fc_layers = self.get_fc_layers(nf)
        self.fc_model = nn.Sequential(*fc_layers).cuda()
        self.save_fc1(nf)
        
        act, val_act = self.activations
        self.fc_data = ClassifierData.from_arrays(self.data.path, 
               (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs)
        self.model=nn.Sequential(*([self.top_model]+fc_layers)).cuda()
        self.freeze()

    def freeze_to(self, n):
        print(f'-- freeze {n}')
        c=children(self.model)[0]
        c=children(c)
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def unfreeze(self): self.freeze_to(0)
    def freeze(self): self.freeze_to(9999)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    def save(self, name): save_model(self.model, self.get_model_path(name))
    def load(self, name): load_model(self.model, self.get_model_path(name))
    
    def train(self, epochs): self.data.train(self.model, epochs)
    def train_fc(self, epochs): self.fc_data.train(self.fc_model, epochs)
    
    @property 
    def lr(self): return self.lr_
        
    @lr.setter
    def lr(self, lr): 
        self.lr_=lr
        set_learning_rate(self.opt, lr)

    @property
    def metrics(self):
        if self.metrics_ is None:
            return [accuracy_multi] if self.data.is_multi else [accuracy]
        return self.metrics_
    
    def fit_gen(self, model, data, epochs, lr, cycle_len=None, metrics=None, callbacks=None):
        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics
        self.opt = self.opt_fn(trainable_params(model), lr=lr)
        if cycle_len is None: self.sched=LossRecorder()
        else: self.sched = CosAnneal(len(data.trn_dl)*cycle_len, init_lr=lr)
        callbacks+=[self.sched]
        self.lr = lr
        for cb in callbacks: cb.on_train_begin(self)
        fit(model, data, epochs, self.crit, self.opt, metrics, callbacks)

    def fit(self, *args, **kwargs):
        self.fit_gen(self.model, self.data, *args, **kwargs)

    def fit_fc(self, *args, **kwargs):
        self.fit_gen(self.fc_model, self.fc_data, *args, **kwargs)
        
    def lr_find(self, is_fc=False, start_lr=1e-5, end_lr=10, patience=20):
        self.save('tmp')
        lrf = LR_Finder(len(self.data.trn_dl), start_lr, end_lr, patience)
        f = self.fit_fc if is_fc else self.fit
        f(1, 1e-4, callbacks=[lrf])
        self.load('tmp')
        return lrf

    def probabilities(self):
        return get_probabilities(self.model, self.data.val_dl)
    