from imports import *
from dataset import *
from fast_gen import *

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def predict_to_bcolz(m, gen, arr, workers=4, verbose=1):
    m._make_predict_function()
    gen.reset()
    lock=threading.Lock()
    try:
        enqueuer = OrderedEnqueuer(gen)
        enqueuer.start(workers=workers, max_queue_size=10)
        batches = enqueuer.get()
        if verbose == 1: progbar = Progbar(target=gen.n_batch)
        for i in range(gen.n_batch):
            batch = next(batches)[0]
            with lock:
                preds = m.predict_on_batch(batch)
                arr.append(preds)
                arr.flush()
            if verbose == 1: progbar.update(i)
    finally: enqueuer.stop()

def freeze_to(m, idx):
    print(f'-- freeze {idx}')
    if isinstance(idx, str): idx=m.layers.index(m.get_layer(idx))
    for layer in m.layers[idx:]: layer.trainable = True
    for layer in m.layers[:idx]: layer.trainable = False
    reset_trainable(m)

def unfreeze(m): freeze_to(m, 0)

def reset_trainable(m): 
    m._collected_trainable_weights = m.trainable_weights
    m.train_function = None
    
def set_trainable(m, name, t): 
    m.get_layer(name).trainable = t
    reset_trainable(m)

def set_lr(m, lr): 
    K.set_value(m.optimizer.lr, lr)
    print(f'-- lr {lr}')

class Modeler():
    def __init__(self, model, ds, tmp_name='tmp', models_name='models', ps=None):
        self.ds,self.ps = ds,ps
        self.tmp_path = os.path.join(self.ds.path, tmp_name)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.models_path = os.path.join(self.ds.path, models_name)
        if not os.path.exists(self.models_path): os.mkdir(self.models_path)
        self.create(model)

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), 
                            chunklen=1, mode='w', rootdir=name)

    def get_activations(self, sz, force=False):
        tmpl = f'_{self.f}_{self.model.input_shape[1]}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val')]
        if os.path.exists(names[0]) and not force: 
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(sz,n) for n in names]
        
    def save_fc1(self, sz):
        self.get_activations(sz)
        if len(self.activations[0]): return
        predict_to_bcolz(self.model, self.ds.fix_it, self.activations[0])
        predict_to_bcolz(self.model, self.ds.val_it, self.activations[1])

    def create_fc_layer(self, nf, actn, name, p):
        res=[]
        if p: res.append(Dropout(p, name=name+'_do'))
        res.append(Dense(nf, activation=actn, name=name))
        return res

    def get_fc_layers(self, nf):
        #res = self.create_fc_layer(nf, 'relu', name='fc1', p=self.ps[0])
        res = self.create_fc_layer(self.ds.c, 'softmax', name='fc2', p=self.ps[0])
        return res

    def add_fc(self, x, layers):
        for l in layers: x = l(x)
        return x
    
    def create(self, model):
        self.model=model
        
        if not self.ps: self.ps=[0]
        elif not isinstance(self.ps, list): self.ps = [self.ps]
            
        nf=self.model.output_shape[1]
        fc_inp = Input((nf,))
        fc_layers = self.get_fc_layers(nf)
        self.fc_model = Model(fc_inp, self.add_fc(fc_inp, fc_layers), 
                              name=self.model.name+'_fc')
        self.save_fc1(nf)
        act, val_act = self.activations
        self.fc_ds = ClassifierDataset.from_arrays(self.ds.path, pass_gen(), pass_gen(), 
               act, self.ds.trn_y, val_act, self.ds.val_y, self.ds.bs)
        
        self.model = Model(self.model.input, 
            self.add_fc(self.model.output, fc_layers), name=self.model.name)
        freeze_to(self.model, -1)
        self.compile()

    def compile(self, lr=0.001, opt=None):
        if opt is None: opt=SGD(lr=lr, momentum=0.9)
        #if opt is None: opt=RMSprop(lr=lr)
        self.model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        self.fc_model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        
    def clear_session(self):
        lr=K.get_value(self.model.optimizer.lr)
        self.save('tmp')
        self.model=None
        self.fc_model=None
        K.clear_session()
        limit_mem()
        self.create()
        self.load('tmp')
        K.set_value(self.model.optimizer.lr, lr)

    def set_lr(self, lr): set_lr(self.model, lr)
    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    def save(self, name): self.model.save_weights(self.get_model_path(name))
    def load(self, name): self.model.load_weights(self.get_model_path(name))
    
    def train(self, epochs): self.ds.train(self.model, epochs)
    def train_fc(self, epochs): self.fc_ds.train(self.fc_model, epochs)
