from imports import *
from torch_imports import *
from fast_gen import *
from pt_models import *
from dataset_pt import *
from sgdr_pt import *


class ConvnetBuilder():
    f_cuts = {
        resnet18:8, resnet34:8, resnet50:8, resnet101:8, resnext50:8,
        wrn:8, dn121:10, inceptionresnet_2:5, inception_4:19
    }

    def __init__(self, f, c, is_multi, ps=None, xtra_fc=None, xtra_cut=0):
        self.f,self.c,self.is_multi,self.ps,self.xtra_cut = f,c,is_multi,ps,xtra_cut
        self.xtra_fc = xtra_fc or []
        
        cut = self.f_cuts[self.f]-xtra_cut
        layers = cut_model(self.f(True), cut)
        self.nf=num_features(layers[-1])*2
        layers += [AdaptiveConcatPool2d(), Flatten]
        self.top_model = nn.Sequential(*layers).cuda()

        n_fc = len(xtra_fc)+1
        if not self.ps: self.ps=[0]*n_fc
        elif not isinstance(self.ps, list): self.ps = [self.ps]*n_fc
            
        fc_layers = self.get_fc_layers()
        self.fc_model = nn.Sequential(*fc_layers).cuda()
        apply_init(self.fc_model, kaiming_normal)

        self.model=nn.Sequential(*([self.top_model]+fc_layers)).cuda()
        #self.model.apply(self.set_mom);

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

            
class Learner():
    def __init__(self, data, models,
                 opt_fn=optim.RMSprop, tmp_name='tmp', models_name='models', metrics=None):
        self.data,self.opt_fn,self.metrics_ = data,opt_fn,metrics
        self.models = models
        self.tmp_path = os.path.join(self.data.path, tmp_name)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.models_path = os.path.join(self.data.path, models_name)
        if not os.path.exists(self.models_path): os.mkdir(self.models_path)
        self.crit = F.binary_cross_entropy if data.is_multi else F.nll_loss
        
        self.save_fc1()
        act, val_act = self.activations
        self.fc_data = ClassifierData.from_arrays(self.data.path, 
               (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs)
        self.freeze()

    @classmethod
    def pretrained_convnet(self, f, data, ps=None, xtra_fc=None, xtra_cut=0, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut)
        return Learner(data, models, **kwargs)
                           
    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)

    def num_features(self): return num_features(self.models.model)
    
    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val')]
        if os.path.exists(names[0]) and not force: 
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf,n) for n in names]
        
    def save_fc1(self):
        self.get_activations()
        if len(self.activations[0]): return
        m=self.models.top_model
        predict_to_bcolz(m, self.data.fix_dl, self.activations[0])
        predict_to_bcolz(m, self.data.val_dl, self.activations[1])

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
    
    def fit_gen(self, model, data, epochs, lr, cycle_len=None, metrics=None, callbacks=None, wd=1e-5):
        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics
        self.opt = self.opt_fn(trainable_params(model), lr=lr, weight_decay=wd)
        self.lr = lr
        if cycle_len is None: self.sched=LossRecorder()
        else: self.sched = CosAnneal(len(data.trn_dl)*cycle_len, init_lr=lr)
        callbacks+=[self.sched]
        for cb in callbacks: cb.on_train_begin(self)
        fit(model, data, epochs, self.crit, self.opt, metrics, callbacks)

    def fit(self, *args, **kwargs):
        self.fit_gen(self.models.model, self.data, *args, **kwargs)

    def fit_fc(self, *args, **kwargs):
        self.fit_gen(self.models.fc_model, self.fc_data, *args, **kwargs)
        
    def lr_find(self, is_fc=False, start_lr=1e-5, end_lr=10):
        self.save('tmp')
        lrf = LR_Finder(len(self.data.trn_dl), start_lr, end_lr)
        f = self.fit_fc if is_fc else self.fit
        f(1, 1e-4, callbacks=[lrf])
        self.load('tmp')
        return lrf

    def probabilities(self):
        return get_probabilities(self.model, self.data.val_dl)
    