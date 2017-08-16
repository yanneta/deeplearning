from imports import *
from fast_gen import *

def read_dirs(path, folder):
    full_path = os.path.join(path, folder)
    all_labels = [os.path.basename(os.path.dirname(f)) 
                  for f in glob(f"{full_path}/*/")]
    fnames = [glob(f"{full_path}/{d}/*.*") for d in all_labels]
    pairs = [(fn,l) for l,f in zip(all_labels, fnames) for fn in f]
    return list(zip(*pairs))+[all_labels]

def n_hot(ids, c):
    res = np.zeros((c,))
    res[ids] = 1
    return res

def folder_source(path, folder):
    fnames, lbls, all_labels = read_dirs(path, folder)
    label2idx = {v:k for k,v in enumerate(all_labels)}
    idxs = [label2idx[lbl] for lbl in lbls]
    c = len(all_labels)
    label_arr = np.stack(n_hot(o, c) for o in idxs)
    return fnames, label_arr, all_labels

class BaseSeq(Sequence):
    def __init__(self, gen_x, gen_y, shuffle, bs, seed=None):
        self.gen_x,self.gen_y,self.bs,self.shuffle = gen_x,gen_y,bs,shuffle
        self.n = self.get_n()
        self.c = self.get_c()
        self.n_batch = math.ceil(self.n/bs)
        self.bs = bs
        self.on_epoch_end()
 
    def __getitem__(self, batch_idx):
        bs = self.bs
        index = batch_idx*bs
        if index+bs > self.n: bs=self.n-index
        idxs = self.idxs[index:index+bs]
        return (self.get(self.gen_x, self.get_x, idxs), 
                self.get(self.gen_y, self.get_y, idxs))

    def __len__(self): return self.n_batch

    def on_epoch_end(self): 
        self.idxs = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
    
    def reset(self): self.on_epoch_end()
        
    def get(self, gen, fn, idxs): return np.stack([gen(fn(i)) for i in idxs])
        
    @abstractmethod
    def get_n(self): raise NotImplementedError
        
    @abstractmethod
    def get_c(self): raise NotImplementedError
        
    @abstractmethod
    def get_x(self, i): raise NotImplementedError
        
    @abstractmethod
    def get_y(self, i): raise NotImplementedError

class BaseArrayIter(BaseSeq):
    def __init__(self, y, gen, shuffle, **kwargs):
        self.y=y
        super().__init__(gen, pass_gen(), shuffle, **kwargs)
    def get_n(self): return len(self.y)
    def get_c(self): return self.y.shape[1]
    
class FilesIter(BaseArrayIter):
    def __init__(self, fnames, y, gen, shuffle, **kwargs):
        self.fnames=fnames
        assert(len(fnames)==len(y))
        super().__init__(y, gen, shuffle, **kwargs)
    def get_x(self, i): return np.array(PIL.Image.open(self.fnames[i]))
    def get_y(self, i): return self.y[i]

class ArraysIter(BaseArrayIter):
    def __init__(self, x, y, gen, shuffle, **kwargs):
        self.x=x
        self.lock=threading.Lock()
        assert(len(x)==len(y))
        super().__init__(y, gen, shuffle, **kwargs)
    def get_x(self, i): 
        with self.lock: return self.x[i]
    def get_y(self, i): 
        with self.lock: return self.y[i]

class Dataset():
    def __init__(self, path, trn_it, fix_it, val_it): 
        self.path,self.trn_it,self.fix_it,self.val_it = path,trn_it,fix_it,val_it
        self.c = self.trn_it.c
    
    @property
    def val_nb(self): return self.val_it.n_batch    
    @property
    def trn_nb(self): return self.trn_it.n_batch
    
    def train(self, m, epochs, **kwargs):
        trn_nb,val_nb = self.trn_nb,self.val_nb
        if epochs<1: trn_nb *= epochs; epochs=1
        return m.fit_generator(self.trn_it, trn_nb, epochs=epochs, workers=4,
           validation_data=self.val_it, validation_steps=val_nb, **kwargs)

class ClassifierDataset(Dataset):
    def __init__(self, path, trn_it, fix_it, val_it, trn_y):
        self.path=path
        self.c=trn_it.c
        self.is_multi = np.all(trn_y.sum(axis=1)==1)
        super().__init__(path, trn_it, fix_it, val_it)
        
    @property
    def bs(self): return self.trn_it.bs
    @property
    def trn_y(self): return self.trn_it.y
    @property
    def val_y(self): return self.val_it.y
        
    @classmethod
    def get_it(self, fn, trn_gen, val_gen, 
               trn_x, trn_y, val_x, val_y, bs):
        return (fn(trn_x, trn_y, gen=trn_gen, bs=bs  , shuffle=True),
                fn(trn_x, trn_y, gen=val_gen, bs=bs//2*3, shuffle=False),
                fn(val_x, val_y, gen=val_gen, bs=bs//2*3, shuffle=False))
        
    @classmethod
    def from_arrays(self, path, trn_gen, val_gen, 
                    trn_x, trn_y, val_x, val_y, bs, **kwargs):
        trn_it,fix_it,val_it = self.get_it(ArraysIter, trn_gen, val_gen,
            trn_x, trn_y, val_x, val_y, bs)
        return self(path, trn_it, fix_it, val_it, trn_y, **kwargs)        

    @classmethod
    def from_paths(self, path, trn_gen, val_gen, bs, 
                   trn_name='train', val_name='val', **kwargs):
        trn_fnames, trn_y, all_labels = folder_source(path, 'train')
        val_fnames, val_y, _ = folder_source(path, 'valid')        
        trn_it,fix_it,val_it = self.get_it(FilesIter, trn_gen, val_gen,
            trn_fnames, trn_y, val_fnames, val_y, bs)
        return self(path, trn_it, fix_it, val_it, trn_y, **kwargs)
