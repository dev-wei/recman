import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={"f": {"format": "%(asctime)s %(levelname)-8s %(message)s"}},
    handlers={
        "h": {"class": "logging.StreamHandler", "formatter": "f", "level": logging.INFO}
    },
    root={"handlers": ["h"], "level": logging.INFO},
)

dictConfig(logging_config)

# #%%
import itertools
import pandas as pd
import tensorflow as tf
from datetime import datetime
import numpy as np
from recman.tf.core.metric import LogLoss, RocAucScore
from tensorboard.plugins.hparams import api as hp
from recman.tf.core import FeatureDictionary, SparseFeat, DenseFeat, MultiValCsvFeat

# from recman import HP_xDeepFM, xDeepFM
from examples.utils import build_features, get_ml_dataset
from recman.tf.hparams import xDeepFM as HyperParams
from recman.tf.core import xDeepFM
from recman.tf.core import TensorBoardLogger

# tf.compat.v1.logging.set_verbosity(20)
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# df_train, df_valid, df_test, domains = get_ml_dataset(frac=0.01)
#%%

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
hp_params["learning_rate"](hp.Discrete([0.01]))
hp_params["optimizer"](hp.Discrete(["adam"]))
#
metrices = (LogLoss(), RocAucScore())

hp_writer = tf.summary.create_file_writer(f"./logs/hparams")
with hp_writer.as_default():
    hp.hparams_config(
        hparams=[hparam.tf_hparam for hparam in hp_params.values()],
        metrics=[
            hp.Metric(f"{prefix}{eval_func}", display_name=f"{prefix}{eval_func}")
            for prefix, eval_func in list(
                itertools.product(*list([["TRAIN_", "VALID_", "TEST_"], metrices]))
            )
        ],
    )

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
ses_num = 0
be_score = None


class BestScoreFinder:
    def __init__(self):
        self.best_score = None

    def __call__(self, variables, eval_results):
        pass
        # eval_results = filter(lambda r: r, eval_results)
        # score = eval_results[-1][0]
        # score = eval_results[0][0] if eval_results[1] is None else eval_results[1][0]
        # if self.best_score is None or score < self.best_score:
        #     print("Best model found!")
        #     self.best_score = score
        #
        #     ckpt = tf.train.Checkpoint(**variables)
        #     ckpt.save(f"./ckpt_model")


best_score_finder = BestScoreFinder()

# exit(1)

for hp_val in hp_params.grid_search():
    # import pickle
    # pickle.dump(hp_val, open("hp", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # model = xDeepFM(feat_dict, hparams=hp_val)
    # print(model.predict(df_X))
    # print(model.variables["linear_w"])
    # model.load()
    # print(model.variables["linear_w"])
    # print(model.predict(df_X))
    # break
    # print(hp_val)
    tb_logger = TensorBoardLogger(hp_params, run_name=run_name, sess_num=ses_num)

    model = xDeepFM(feat_dict, hp_val, metrics=metrices, epoch=10)
    model.fit(
        df_X,
        df_X["LABEL"].values,
        # X_valid=df_valid,
        # y_valid=df_valid["label"].values,
        tb_logger=tb_logger,
        epoch_callback=best_score_finder,
        random_seed_for_mini_batch=False,
    )
    ses_num += 1
# print(model.predict(df_X))
