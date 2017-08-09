import PIL, os, numpy as np, math, matplotlib.pyplot as plt, collections
from keras import utils
from keras.preprocessing import image
from abc import abstractmethod
from glob import glob

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
    fnames, lbls, all_labels = read_dirs(path, 'train')
    label2idx = {v:k for k,v in enumerate(all_labels)}
    idxs = [label2idx[lbl] for lbl in lbls]
    c = len(all_labels)
    label_arr = np.stack(n_hot(o, c) for o in idxs)
    return fnames, label_arr, all_labels

def apply_gen(im, targ_r, scale_fn, fns):
    for fn in fns: im=fn(im)
    return scale_fn(im, targ_r)

def base_gen(targ_r, scale_fn, fns=[]): 
    if not isinstance(fns, collections.Iterable): fns=[fns]
    return lambda im: apply_gen(im, targ_r, scale_fn, fns)

def pass_gen(): return lambda o: o

class BaseIter(image.Iterator):
    def __init__(self, gen_x, gen_y, bs=64, shuffle=False, seed=None):
        self.gen_x,self.gen_y,self.bs,self.shuffle = gen_x,gen_y,bs,shuffle
        self.samples = self.get_n()
        self.n_batch = math.ceil(self.samples/bs)
        super().__init__(self.samples, bs, shuffle, seed)
        
    @abstractmethod
    def get_n(self): raise NotImplementedError
        
    @abstractmethod
    def get_x(self, i): raise NntImplementedError
        
    @abstractmethod
    def get_y(self, i): raise NotImplementedError
        
    def get(self, gen, fn, idxs): return np.stack([gen(fn(i)) for i in idxs])
    
    def next(self):
        with self.lock: idxs, curr_idx, bs = next(self.index_generator)
        return (self.get(self.gen_x, self.get_x, idxs), 
                self.get(self.gen_y, self.get_y, idxs))
    

class FilesIter(BaseIter):
    def __init__(self, fnames, y, gen, bs=64, shuffle=False, seed=None):
        self.fnames,self.y=fnames,y
        assert(len(fnames)==len(y))
        super().__init__(gen, pass_gen(), bs, shuffle, seed)
        
    def get_x(self, i): return PIL.Image.open(self.fnames[i])
    def get_y(self, i): return self.y[i]
    def get_n(self): return len(self.y)

def center_crop(im):
    r,c,_ = im.shape
    min_s = min(r,c)
    start_r = math.ceil((r-min_s)/2)
    start_c = math.ceil((c-min_s)/2)
    return im[start_r:start_r+min_s, start_c:start_c+min_s]

def scale_to(x, ratio, targ): return max(math.floor(x*ratio), targ)
    
def scale_min(im, targ_r):
    r,c = im.size
    ratio = targ_r/min(r,c)
    sz = (scale_to(r, ratio, targ_r), scale_to(c, ratio, targ_r))
    return np.array(im.resize(sz, PIL.Image.BILINEAR))

def scale_and_center(im, targ_r): return center_crop(scale_min(im, targ_r))
