from keras.callbacks import *

class LossRecorder(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.iterations = 0.

    @property
    def lr(self): return self.model.optimizer.lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.losses = []
    
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.iterations += 1
        self.history.setdefault('lr', []).append(K.get_value(self.lr))
        self.history.setdefault('iterations', []).append(self.iterations)
        for k, v in logs.items(): self.history.setdefault(k, []).append(v)
        self.losses.append(logs.get('loss'))
    
class LR_Finder(LossRecorder):
    def __init__(self, nb, start_lr=1e-5, end_lr=10, patience=20, loss_mom=0.9):
        super().__init__()
        self.lr_mult = (end_lr/start_lr)**(1/nb)
        self.start_lr,self.patience,self.loss_mom = start_lr,patience,loss_mom

    def incr_lr(self):
        return self.start_lr * (self.lr_mult**self.iterations)
    
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        self.best=1e9
        self.best_iters=0
        self.avg_loss = None
        K.set_value(self.lr, self.incr_lr())
            
    def on_batch_end(self, epoch, logs=None):
        super().on_batch_end(epoch, logs)
        K.set_value(self.lr, self.incr_lr())
        loss=logs.get('loss')
        if self.avg_loss is None: self.avg_loss=loss
        else: self.avg_loss = self.avg_loss*self.loss_mom + loss*(1-self.loss_mom)
        self.losses[-1]=self.avg_loss
        if self.iterations<10: return
        
        if self.avg_loss>self.best*4:
            self.best_iters+=1
            if self.best_iters>self.patience:
                self.model.stop_training = True
        else:
            self.best_iters=0
            if self.avg_loss<self.best: self.best=self.avg_loss

class CosAnneal(LossRecorder):
    def __init__(self, nb, init_lr=0.1):
        super().__init__()
        self.check = nb
        self.init_lr = init_lr

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        K.set_value(self.lr, self.init_lr)
    
    def on_batch_end(self, epoch, logs=None):
        super().on_batch_end(epoch, logs)
        K.set_value(self.lr, self.cos_anneal())

    def cos_anneal(self):
        cos_inner = np.pi * ((self.iterations-2) % (self.check))
        cos_out = np.cos(cos_inner/self.check) + 1
        return float(self.init_lr / 2 * cos_out)