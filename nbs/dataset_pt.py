from imports import *
from torch_imports import *
from fast_gen import *

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
inception_stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)

def resize_img(fname, targ, path, new_path):
    dest = os.path.join(path,new_path,str(targ),fname)
    if os.path.exists(dest): return
    im = PIL.Image.open(os.path.join(path, fname)).convert('RGB')
    r,c = im.size
    ratio = targ/min(r,c)
    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    im.resize(sz, PIL.Image.LINEAR).save(dest)

def resize_imgs(fnames, targ, path, new_path):
    with ThreadPoolExecutor(8) as e:
        ims = e.map(lambda x: resize_img(x, targ, path, 'tmp'), fnames)
        for x in tqdm(ims, total=len(fnames)): pass
    return os.path.join(path,new_path,str(targ))
        
def read_dirs(path, folder):
    full_path = os.path.join(path, folder)
    all_labels = [os.path.basename(os.path.dirname(f)) 
                  for f in iglob(f"{full_path}/*/")]
    fnames = [iglob(f"{full_path}/{d}/*.*") for d in all_labels]
    pairs = [(os.path.relpath(fn,path), l) for l,f in zip(all_labels, fnames) for fn in f]
    return list(zip(*pairs))+[all_labels]

def n_hot(ids, c):
    res = np.zeros((c,), dtype=np.float32)
    res[ids] = 1
    return res

def folder_source(path, folder):
    fnames, lbls, all_labels = read_dirs(path, folder)
    label2idx = {v:k for k,v in enumerate(all_labels)}
    idxs = [label2idx[lbl] for lbl in lbls]
    c = len(all_labels)
    label_arr = np.array(idxs, dtype=int)
    return fnames, label_arr, all_labels

def parse_csv_labels(fn, skip_header=True):
    skip = 1 if skip_header else 0
    csv_lines = [o.strip().split(',') for o in open(fn)][skip:]
    csv_labels = {a:b.split(' ') for a,b in csv_lines}
    all_labels = list(set(p for o in csv_labels.values() for p in o))
    label2idx = {v:k for k,v in enumerate(all_labels)}
    return sorted(csv_labels.keys()), csv_labels, all_labels, label2idx

def nhot_labels(label2idx, csv_labels, fnames, c):
    all_idx = {k: n_hot([label2idx[o] for o in v], c) 
               for k,v in csv_labels.items()}
    return np.stack([all_idx[o] for o in fnames])

def csv_source(folder, csv_file, skip_header=True, suffix=''):
    fnames,csv_labels,all_labels,label2idx = parse_csv_labels(
        csv_file, skip_header)
    label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
    full_names = [fn+suffix for fn in fnames]
    is_single = np.all(label_arr.sum(axis=1)==1)
    if is_single: label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels

class BaseDataset(Dataset):
    def __init__(self, transform, target_transform):
        self.transform,self.target_transform = transform,target_transform
        self.lock=threading.Lock()
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()
 
    def __getitem__(self, idx):
        return (self.get(self.transform, self.get_x, idx), 
                self.get(self.target_transform, self.get_y, idx))

    def __len__(self): return self.n
        
    def get(self, tfm, fn, idx): 
        return fn(idx) if tfm is None else tfm(fn(idx))
        
    @abstractmethod
    def get_n(self): raise NotImplementedError
    @abstractmethod
    def get_c(self): raise NotImplementedError
    @abstractmethod
    def get_sz(self): raise NotImplementedError
    @abstractmethod
    def get_x(self, i): raise NotImplementedError
    @abstractmethod
    def get_y(self, i): raise NotImplementedError
    @property
    def is_multi(self): return False

    
class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path,self.fnames = path,fnames
        super().__init__(transform, None)
    def get_n(self): return len(self.y)
    def get_sz(self): return self.transform.sz
    def get_x(self, i): 
        im = PIL.Image.open(os.path.join(self.path, self.fnames[i])).convert('RGB')
        return np.array(im, dtype=np.float32)/255.
    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

            
class FilesArrayDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return self.y[i]

    
class FilesIndexArrayDataset(FilesArrayDataset):
    def get_c(self): return int(self.y.max())+1

    
class FilesNhotArrayDataset(FilesArrayDataset):
    def get_c(self): return self.y.shape[1]
    @property
    def is_multi(self): return True

    
class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x,self.y=x,y
        assert(len(x)==len(y))
        super().__init__(transform, None)
    def get_x(self, i): 
        with self.lock: return self.x[i]
    def get_y(self, i): 
        with self.lock: return self.y[i]
    def get_n(self): return len(self.y)
    def get_sz(self): return self.x.shape[1]

    
class ArraysIndexDataset(ArraysDataset):
    def get_c(self): return int(self.y.max())+1

    
class ArraysNhotDataset(ArraysDataset):
    def get_c(self): return self.y.shape[1]
    @property
    def is_multi(self): return True

    
class ModelData():
    def __init__(self, path, datasets, bs, num_workers): 
        trn_ds,val_ds = datasets
        self.path,self.bs,self.num_workers = path,bs,num_workers
        self.trn_dl,self.fix_dl,self.val_dl = [self.get_dl(ds,shuf) 
            for ds,shuf in [(trn_ds,True),(trn_ds,False),(val_ds,False)]]

    def get_dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True)

    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def sz(self): return self.trn_ds.sz
    @property
    def c(self): return self.trn_ds.c
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y

    def resize(self, targ, new_path):
        ds = self.trn_ds.resize_imgs(targ,new_path), self.val_ds.resize_imgs(targ,new_path)
        return self.__class__(ds[0].path, ds, self.bs, self.num_workers)

    
class ClassifierData(ModelData):
    @property
    def is_multi(self): return self.trn_dl.dataset.is_multi
    
    def tfms_from_model(f_model, sz, aug_tfms=[], max_zoom=None, pad=0):
        stats = inception_stats if f_model in inception_models else imagenet_stats
        tfm_norm = Normalize(*stats)
        val_tfm = image_gen(tfm_norm, sz, pad=pad)
        trn_tfm=image_gen(tfm_norm, sz, tfms=aug_tfms, max_zoom=max_zoom, pad=pad)
        return trn_tfm, val_tfm
        
    @classmethod
    def get_ds(self, fn, trn, val, tfms, **kwargs):
        return (fn(trn[0], trn[1], tfms[0], **kwargs),
                fn(val[0], val[1], tfms[1], **kwargs))
        
    @classmethod
    def from_arrays(self, path, trn, val, bs, tfms=(None,None), num_workers=4):
        datasets = self.get_ds(ArraysIndexDataset, trn, val, tfms)
        return self(path, datasets, bs, num_workers)

    @classmethod
    def from_paths(self, path, bs, tfms, trn_name='train', val_name='val', num_workers=4):
        trn,val = [folder_source(path, o) for o in ('train', 'valid')]
        datasets = self.get_ds(FilesIndexArrayDataset, trn, val, tfms, path=path)
        return self(path, datasets, bs, num_workers)

    @classmethod
    def from_csv(self, path, csv_fname, bs, tfms,
               val_idxs=None, suffix='', skip_header=True, num_workers=4): 
        fnames,y,_ = csv_source(path, csv_fname, skip_header, suffix)

        val_idxs,fnames = np.array(val_idxs),np.array(fnames)
        val = fnames[val_idxs],y[val_idxs]
        mask = np.zeros(len(fnames),dtype=bool)
        mask[val_idxs] = True
        trn = fnames[~mask],y[~mask]
        
        f = FilesIndexArrayDataset if len(trn[1].shape)==1 else FilesNhotArrayDataset
        datasets = self.get_ds(f, trn, val, tfms, path=path)
        return self(path, datasets, bs, num_workers)
