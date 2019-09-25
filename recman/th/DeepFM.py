import tensorflow as tf
import numpy as np
import tqdm
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.utils import shuffle
from .input import SparseFeat, DenseFeat, MultiValFeat


class DeepFM(BaseEstimator, TransformerMixin):
    pass