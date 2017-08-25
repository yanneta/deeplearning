from imports import *
from torch_imports import *
from dataset_pt import *

class ColumnarDataset(Dataset):
    def __init__(self,*args):
        *xs,y=args
        self.xs = [T(x) for x in xs]
        self.y = T(y).unsqueeze(1)
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return [o[idx] for o in self.xs] + [self.y[idx]]
    
    @classmethod
    def from_data_frame(self, df, cols_x, col_y):
        cols = [df[o] for o in cols_x+[col_y]]
        return self(*cols)

class ColumnarModelData(ModelData):
    def __init__(self, trn_ds, val_ds, bs): 
        super().__init__(DataLoader(trn_ds, bs, shuffle=True),
            DataLoader(val_ds, bs*2, shuffle=False))

    @classmethod
    def from_data_frames(self, trn_df, val_df, cols_x, col_y, bs):
        return self(ColumnarDataset.from_data_frame(trn_df, cols_x, col_y),
                    ColumnarDataset.from_data_frame(val_df, cols_x, col_y), bs)
    
    @classmethod
    def from_data_frame(self, val_idxs, df, cols_x, col_y, bs):
        ((val_df, trn_df),) = split_by_idx(val_idxs, df)
        return self.from_data_frames(trn_df, val_df, cols_x, col_y, bs)

    
class CollabFilter(Dataset):
    def __init__(self, user_col, item_col, ratings):
        self.ratings = ratings
        self.n = len(ratings)
        (self.users,self.user2idx,self.user_col,self.n_users) = self.proc_col(user_col)
        (self.items,self.item2idx,self.item_col,self.n_items) = self.proc_col(item_col)
        self.min_score,self.max_score = min(ratings),max(ratings)
        self.cols = [self.user_col,self.item_col,self.ratings]

    def proc_col(self,col):
        uniq = col.unique()
        name2idx = {o:i for i,o in enumerate(uniq)}
        return (uniq, name2idx, [name2idx[x] for x in col], len(uniq))
        
    def __len__(self): return self.n
    def __getitem__(self, idx): return [o[idx] for o in self.cols]

    def to_model_data(self, val_idxs, bs):
        val, trn = zip(*split_by_idx(val_idxs, *self.cols))
        return ColumnarModelData(ColumnarDataset(*trn),ColumnarDataset(*val),bs)
    
    def get_model(self, n_factors):
        return EmbeddingDotBias(n_factors, self.n_users, self.n_items, self.min_score, self.max_score)


def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.05,0.05)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self, n_factors, n_users, n_items, min_score, max_score):
        super().__init__()
        self.min_score,self.max_score = min_score,max_score
        (self.u, self.i, self.ub, self.ib) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]
        
    def forward(self, users, items):
        um = self.u(users)* self.i(items)
        res = um.sum(1) + self.ub(users).squeeze() + self.ib(items).squeeze()
        return F.sigmoid(res) * (self.max_score-self.min_score) + self.min_score
