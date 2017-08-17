from imports import *

class Callback:
    def on_train_begin(self, learner): pass
    def on_epoch_end(self, metrics): pass
    def on_batch_end(self, metrics): pass

class LossRecorder(Callback):
    def on_train_begin(self, learner): 
        self.learner=learner
        self.losses,self.lrs,self.iterations = [],[],[]
        self.iteration = 0
        self.epoch = 0
        
    def on_epoch_end(self, metrics):
        self.epoch += 1
    
    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.learner.lr)
        self.iterations.append(self.iteration)
        self.losses.append(loss)

    def plot_loss(self):
        plt.plot(self.iterations, self.losses)

    def plot_lr(self):
        plt.plot(self.iterations, self.lrs)

        
class LR_Finder(LossRecorder):
    def __init__(self, nb, start_lr=1e-7, end_lr=10):
        super().__init__()
        self.lr_mult = (end_lr/start_lr)**(1/nb)
        self.start_lr = start_lr

    def incr_lr(self):
        return self.start_lr * (self.lr_mult**self.iteration)
    
    def on_train_begin(self, learner):
        super().on_train_begin(learner)
        self.best=1e9
        learner.lr = self.incr_lr()
            
    def on_batch_end(self, loss):
        super().on_batch_end(loss)
        if self.iteration<10: return
        if math.isnan(loss) or loss>self.best*4: return True
        self.learner.lr = self.incr_lr()
        self.losses[-1]=loss
        if loss<self.best: self.best=loss

    def plot(self):
        plt.plot(self.lrs[:-5], self.losses[:-5])
        plt.xscale('log')

class CosAnneal(LossRecorder):
    def __init__(self, nb, init_lr=0.1):
        super().__init__()
        self.check = nb
        self.init_lr = init_lr

    def on_train_begin(self, learner):
        super().on_train_begin(learner)
        learner.lr = self.init_lr/100.
    
    def on_batch_end(self, loss):
        super().on_batch_end(loss)
        self.learner.lr = self.cos_anneal()

    def cos_anneal(self):
        cos_inner = np.pi * ((self.iteration-2) % (self.check))
        cos_out = np.cos(cos_inner/self.check) + 1
        return float(self.init_lr / 2 * cos_out)
    