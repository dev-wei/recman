import logging
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from recman.tf.BestModelFinder import BestModelFinder
from recman.tf.core import TensorBoardLogger
from recman.tf.core import xDeepFM
from recman.tf.core.metric import LogLoss, RocAucScore
from recman.tf.hparams import xDeepFM as HyperParams
from recman.tf.inputs import FeatureDictionary, MultiValCsvFeat, SparseFeat

log = logging.getLogger(__name__)
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

RANDOM_SEED = 2019
TB_LOG_DIR = "./logs"

# fmt: off
df_X = pd.DataFrame(
    [
        ["USER_1", "ITEM_1", ["Treadmill", 3], "Outdoor", 123, "a|b|d", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_1", "ITEM_4", ["Treadmill", 3], "Treadmill", 123, "a|b", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_1", "ITEM_3", ["Outdoor", 3], "Outdoor", 124, "a|b", ["a", "b"], ["ITEM_1", "ITEM_2", "ITEM_3"], 1, ],
        ["USER_1", "ITEM_5", ["Outdoor", 3], "Outdoor", 124, "a|b", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_4", "ITEM_6", ["Rest", 3], "Rest", 124, "a|b", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
        ["USER_2", "ITEM_1", ["Treadmill", 3], "Treadmill", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
        ["USER_2", "ITEM_4", ["Treadmill", 3], "Treadmill", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2", "ITEM_3"], 0, ],
        ["USER_2", "ITEM_2", ["Outdoor", 3], "Outdoor", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_2", "ITEM_5", ["Outdoor", 3], "Outdoor", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_5", "ITEM_1", ["Treadmill", 3], "Treadmill", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
        ["USER_5", "ITEM_3", ["Rest", 3], "Rest", 125, "b|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_3", "ITEM_1", ["Treadmill", 3], "Treadmill", 125, "a|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_3", "ITEM_4", ["Treadmill", 3], "Treadmill", 125, "a|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 1, ],
        ["USER_3", "ITEM_2", ["Outdoor", 3], "Outdoor", 125, "a|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
        ["USER_6", "ITEM_2", ["Outdoor", 3], "Outdoor", 125, "a|c", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
        ["USER_6", "ITEM_5", ["Outdoor", 3], "Outdoor", 125, "a|b|c|d", ["a", "b"], ["ITEM_1", "ITEM_2"], 0, ],
    ],
    columns=["USER_ID", "CLASS_ID", "CATEGORY_COUNT", "CATEGORY", "TIMESTAMP", "HISTORICAL_CATEGORIES", "MULTI_VAL_1",
             "SEQUENCE_VAL_1", "LABEL", ], )
# fmt: on

feat_dict = FeatureDictionary()
feat_dict["USER_ID"] = SparseFeat(
    name="USER_ID",
    feat_size=len(np.unique(df_X.USER_ID.values)),
    dtype=tf.int64,
    description="0 presents new user",
)
feat_dict["CLASS_ID"] = SparseFeat(
    name="CLASS_ID",
    feat_size=len(np.unique(df_X.CLASS_ID.values)),
    dtype=tf.int64,
    description="0 presents the first class",
)
feat_dict["CATEGORY"] = SparseFeat(
    name="CATEGORY",
    feat_size=len(np.unique(df_X.CATEGORY.values)),
    dtype=tf.int64,
    # weights={"Outdoor": -0.5},
    description="0 presents the first category",
)
feat_dict["HISTORICAL_CATEGORIES"] = MultiValCsvFeat(
    name="HISTORICAL_CATEGORIES",
    tags=("a", "b", "c", "d"),
    dtype=tf.string,
    # weights={"a": -0.5, "d": -1},
    description="workout categories a user used to engage with",
)
feat_dict.initialize(df_X)

hp_params = HyperParams()
hp_params[HyperParams.LearningRate](hp.Discrete([0.01]))
hp_params[HyperParams.Optimizer](hp.Discrete(["adam"]))

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
        df_X,
        df_X["LABEL"].values,
        # X_valid=df_valid,
        # y_valid=df_valid["label"].values,
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
BestHpVal: {best_model_finder.best_model.hparams}
BestEvalResults: {best_model_finder.best_eval_results}
---------------------------------
"""
)

df_test = df_X.copy()
df_test["PRED"] = best_model_finder.best_model.predict(df_test)
df_test.sort_values("PRED", ascending=False)
log.info(df_test)