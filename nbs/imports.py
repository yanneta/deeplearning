import PIL, os, numpy as np, math, collections, cv2, threading, json, bcolz, random, scipy
import matplotlib.pyplot as plt
from abc import abstractmethod
from glob import glob

import random, pandas as pd, pickle, sys
from itertools import chain

def in_notebook(): return 'ipykernel' in sys.modules

from tqdm import trange, tqdm, tqdm_notebook, tnrange
if in_notebook(): trange=tnrange; tqdm=tqdm_notebook