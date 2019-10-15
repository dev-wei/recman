import pandas as pd
import logging
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from recman.examples.utils import create_ml_features, get_ml_dataset
from recman.tf.BestModelFinder import BestModelFinder
from recman.tf.core import TensorBoardLogger
from recman.tf.core import xDeepFM
from recman.tf.core.metric import LogLoss, RocAucScore
from recman.tf.hparams import xDeepFM as HyperParams


log = logging.getLogger(__name__)
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

RANDOM_SEED = 2019
TB_LOG_DIR = "./logs"

df_train, df_valid, df_test, domains = get_ml_dataset(frac=0.7)
df_all = pd.concat([df_train, df_valid, df_test], axis=0)
feat_dict = create_ml_features(df_all, domains)

hp_params = HyperParams()
hp_params["learning_rate"](hp.Discrete([0.01, 0.005]))
hp_params["optimizer"](hp.Discrete(["adam"]))

metrices = (LogLoss(), RocAucScore())
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
best_model_finder = BestModelFinder()

for sess_num, hp_val in enumerate(hp_params.grid_search()):
    tb_logger = TensorBoardLogger(
        hp_params, run_name=run_name, sess_num=sess_num, log_dir=TB_LOG_DIR
    )
    model = xDeepFM(
        feat_dict,
        hp_val,
        batch_size=128,
        metrics=metrices,
        random_seed=RANDOM_SEED,
        epoch=1,
    )
    model.fit(
        X_train=df_train,
        y_train=df_train["label"].values,
        X_valid=df_valid,
        y_valid=df_valid["label"].values,
        tb_logger=tb_logger,
        epoch_callback=best_model_finder,
        random_seed_for_mini_batch=False,
    )



log.info(
    f"""
---------------------------------
Training job has completed
RunName: {run_name}
BestScore: {best_model_finder.best_score}
BestHpVal: {best_model_finder.best_hp_val}
BestEvalResults: {best_model_finder.best_eval_results}
---------------------------------
"""
)
