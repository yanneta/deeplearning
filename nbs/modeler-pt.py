from imports import *
from dataset import *
from fast_gen import *

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
