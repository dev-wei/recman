# #%%
import itertools
import pandas as pd
import tensorflow as tf
from datetime import datetime

from sklearn.metrics import *
from tensorboard.plugins.hparams import api as hp
from examples.utils import build_features, get_ml_dataset
from recman.tf.core import TensorBoardLogger
from recman.tf.core.metric import LogLoss, RocAucScore

# from recman import HP_xDeepFM, xDeepFM
# from recman.utils import TensorBoardCallback
from recman.tf.hparams import xDeepFM as HyperParams
from recman.tf.core import xDeepFM

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

df_train, df_valid, df_test, domains = get_ml_dataset(frac=0.7)
df_all = pd.concat([df_train, df_valid, df_test], axis=0)
feat_dict = build_features(df_all, domains)
#
#%%

hparams = HyperParams()
hparams["learning_rate"](hp.Discrete([0.01, 0.005]))
hparams["optimizer"](hp.Discrete(["adam"]))
metrices = (LogLoss(), RocAucScore())

session_num = 0
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
for hp_val in hparams.grid_search():
    tb_logger = TensorBoardLogger(
        hparams, metrices, run_name=run_name, sess_num=session_num
    )
    model = xDeepFM(feat_dict, hp_val, metrices=metrices)
    model.fit(
        X_train=df_train,
        y_train=df_train["label"].values,
        X_valid=df_valid,
        y_valid=df_valid["label"].values,
        tb_logger=tb_logger,
    )
    session_num += 1
