import os, json, numpy as np, math, bcolz
from glob import glob

from keras import backend as K
from keras.layers import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Lambda, Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

from keras.utils.generic_utils import Progbar
from keras.utils.data_utils import GeneratorEnqueuer

def predict_to_bcolz(m, generator, preds, labels, workers=2, verbose=0):
    m._make_predict_function()
    steps=math.ceil(generator.n/generator.batch_size)
    
    try:
        enqueuer = GeneratorEnqueuer(generator, wait_time=0.01)
        enqueuer.start(workers=workers, max_queue_size=10)
        batches = enqueuer.get()
        if verbose == 1: progbar = Progbar(target=steps)

        for i in range(steps):
            batch = next(batches)
            x = batch[0]; y = batch[1]
            preds.append(m.predict_on_batch(x))
            labels.append(y)
            if verbose == 1: progbar.update(i)
        preds.flush()
        labels.flush()

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

class Vgg16():
    """
        The VGG 16 Imagenet model
    """
    vgg_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1,1,3))
    dense_layers = ['fc1', 'fc2', 'predictions']
    pre_bn_layers = ['flatten', 'fc1', 'fc2']

    def __init__(self, path=None, bs=48, vert_flip=False, ps=None, bn_layers=None,
            trn_name='train', val_name='valid', tmp_name='tmp', models_name='models',
            gen=image.ImageDataGenerator(horizontal_flip=True, rotation_range=5, 
                height_shift_range=0.05, width_shift_range=0.05,
                channel_shift_range=0.05)):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.path,self.bs,self.ps,self.bn_layers = path,bs,ps,bn_layers
        self.bs=bs
        if path: 
            if not gen.preprocessing_function:
                gen.preprocessing_function=self.preprocess
            if vert_flip: gen.vertical_flip=True
            self.gen=gen
            self.setup_path(trn_name, val_name, tmp_name, models_name)
        else:
            self.model = VGG16()
        self.get_classes()
        
    def setup_path(self, trn_name, val_name, tmp_name, models_name):
        self.trn_path = os.path.join(self.path, trn_name)
        self.val_path = os.path.join(self.path, val_name)
        self.tmp_path = os.path.join(self.path, tmp_name)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.models_path = os.path.join(self.path, models_name)
        if not os.path.exists(self.models_path): os.mkdir(self.models_path)
        self.val_gen=image.ImageDataGenerator(preprocessing_function=self.preprocess)
        self.batches = self.get_batches()
        self.val_batches = self.get_batches(shuffle=False, is_trn=False)
        self.nb = math.ceil(self.batches.n/self.batches.batch_size)
        self.val_nb = math.ceil(self.val_batches.n/self.val_batches.batch_size)
        self.nc=self.batches.num_class
        self.create()

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), 
                            chunklen=1, mode='w', rootdir=name)

    def get_fc1_arrays(self, force=False):
        names = [os.path.join(self.tmp_path, p+'.bc') for p in (
                'arr', 'y_arr', 'arr_val', 'y_arr_val')]
        if os.path.exists(names[0]) and not force: 
            arrs = [bcolz.open(p) for p in names]
        else:
            sizes = (4096, self.nc, 4096, self.nc) # 25088
            arrs = [self.create_empty_bcolz(s,p) for s,p in zip(sizes, names)]
        (self.arr, self.y_arr, self.arr_val, self.y_arr_val) = arrs
        
    def save_fc1(self):
        self.get_fc_model()
        self.get_fc1_arrays()
        if len(self.arr): return
        trn_batches = self.get_batches(shuffle=False, is_trn=True)
        predict_to_bcolz(self.conv_model, trn_batches, self.arr, self.y_arr, verbose=1)
        predict_to_bcolz(self.conv_model, 
                         self.val_batches, self.arr_val, self.y_arr_val, verbose=1)

    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache.
        """
        if self.batches:
            self.class_idx = self.batches.class_indices
            self.classes = [0]*self.nc
            for label,idx in self.class_idx.items(): self.classes[idx]=label
        else:
            fname = 'imagenet_class_index.json'
            fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
            with open(fpath) as f: self.class_idx = json.load(f)
            self.nc = len(self.class_idx)
            self.classes = [self.class_idx[str(i)][1] for i in range(self.nc)]

    def preprocess(self, x):
        """
            Subtracts the mean RGB value, and transposes RGB to BGR.
            The mean RGB was computed on the image set used to train the VGG model.
            (Similar to .applications.imagenet_utils.preprocess_input)

            Args: 
                x: Image array (height x width x channels)
            Returns:
                Image array (height x width x transposed_channels)
        """
        return x[:, :, ::-1] - self.vgg_mean   # reverse axis rgb->bgr

    
    def get_path(self, is_trn): 
        return self.trn_path if is_trn else self.val_path
    
    def get_batches(self, shuffle=True, is_trn=True, class_mode='categorical'):
        """
            Generates batches of augmented/normalized data.
            Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        gen = self.gen if shuffle else self.val_gen
        return gen.flow_from_directory(self.get_path(is_trn), target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=self.bs)

    def get_bn_params(self):
        m=self.model
        batches = self.get_batches()
        x,y = next(batches)
        ms=[Model(m.input, m.get_layer(o).output) for o in self.pre_bn_layers]
        bn_inp = [m.predict_on_batch(x) for m in ms]
        means = [o.mean(0) for o in bn_inp]
        stds = [o.var(0) for o in bn_inp]
        bn_params = [(np.sqrt(s),m,m,s) for m,s in zip(means,stds)]
        for idx,keep in enumerate(self.bn_layers):
            if not keep: bn_params[idx]=0
        return bn_params

    def create_fc(self, x, nf, actn, name, p, bn, set_weights=True):
        if bn:
            bnl = BatchNormalization(name=name+'_bn')
            x = bnl(x)
            bnl.set_weights(bn)
        if p: x = Dropout(p, name=name+'_do')(x)
        fc = Dense(nf, activation=actn, name=name)
        x = fc(x)
        if set_weights: 
            fc.set_weights(self.model.get_layer(name).get_weights())
        return x
    
    def get_fc_model(self):
        m=self.model
        fc_start = m.get_layer('fc1')
        fc_start_idx = m.layers.index(fc_start)
        self.conv_model = Model(m.input, fc_start.output)
        fc_inp = Input(batch_shape=self.conv_model.output_shape)
        x=fc_inp
        fc_layers = m.layers[fc_start_idx+1:]
        for l in fc_layers: x=l(x)
        self.fc_model = Model(fc_inp, x)
    
    def create(self):
        self.model = VGG16()
        m=self.model
        if not self.ps: self.ps=[0]*3
        elif not isinstance(self.ps, list): self.ps = [0,0,self.ps]
        assert(len(self.ps)==3)

        if self.bn_layers: bn=self.get_bn_params()
        else: bn=[0]*3
        fl = m.get_layer('flatten')
        x = fl.output

        x = self.create_fc(x, 4096, 'relu', name='fc1', p=self.ps[0], bn=bn[0])
        x = self.create_fc(x, 4096, 'relu', 'fc2', self.ps[1], bn[1])
        x = self.create_fc(x, self.nc, 'softmax', 'predictions', 
                           self.ps[2], bn[2], set_weights=False)

        m.get_layer('fc1').input_nodes=None

        self.model = Model(m.input, x)
        self.save_fc1()
        freeze_to(self.model, -1)
        self.compile()


    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes

    def compile(self, lr=0.001, opt=None):
        """
            Configures the model for training.
        """
        if opt is None: opt=SGD(lr=lr, momentum=0.9)
        self.model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        self.fc_model.compile(opt, 'categorical_crossentropy', ['accuracy'])


    def fit(self, epochs=1, workers=2, callbacks=None):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
        """
        nb = self.nb
        if epochs<1: nb*=epochs; epochs=1
        self.model.fit_generator(self.batches, steps_per_epoch=nb,
            epochs=epochs, validation_data=self.val_batches, validation_steps=self.val_nb,
            workers=workers, callbacks=callbacks)
        
    def fit_fc(self, epochs=1, callbacks=None):
        nb = self.nb
        if epochs<1: nb*=epochs; epochs=1
        self.fc_model.fit(self.arr, self.y_arr, epochs=epochs, 
            validation_data=(self.arr_val, self.y_arr_val), callbacks=callbacks)
        
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