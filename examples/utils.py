import os
import numpy as np
import tensorflow as tf
from examples.datasets.ml_100k import get_data
from recman.tf.core import FeatureDictionary, SparseFeat, DenseFeat, MultiValCsvFeat
from sklearn.preprocessing import MinMaxScaler


def get_ml_dataset(frac=0.5):
    current_dir = os.path.dirname(os.path.realpath("__file__")) + "/data"
    df_all, df_test, domains = get_data(current_dir)
    df_all, df_test = df_all.sample(frac=frac), df_test

    df_all.loc[df_all.rating < 4, "label"] = 0
    df_all.loc[df_all.rating >= 4, "label"] = 1
    df_test.loc[df_test.rating < 4, "label"] = 0
    df_test.loc[df_test.rating >= 4, "label"] = 1
    df_train = df_all.sample(frac=0.5, random_state=2019)
    df_valid = df_all.drop(df_train.index)

    df_all.info()
    print(df_all[df_all.rating < 4].shape)
    print(df_all[df_all.rating >= 4].shape)

    df_test.info()
    return df_train, df_valid, df_test, domains


def build_features(df_data, domains):
    # fmt: off
    feat_dict = FeatureDictionary()
    feat_dict["user_id"] = SparseFeat(
        name="user_id",
        feat_size=len(np.unique(df_data.user_id.values)),
        dtype=tf.int64
    )
    feat_dict["item_id"] = SparseFeat(
        name="item_id",
        feat_size=len(np.unique(df_data.item_id.values)),
        dtype=tf.int64
    )
    feat_dict["gender"] = SparseFeat(
        name="gender",
        feat_size=len(np.unique(df_data.gender.values)),
        dtype=tf.int64
    )
    feat_dict["occupation"] = SparseFeat(
        name="occupation",
        feat_size=len(np.unique(df_data.occupation.values)),
        dtype=tf.int64,
    )
    feat_dict["zip"] = SparseFeat(
        name="zip",
        feat_size=len(np.unique(df_data.zip.values)),
        dtype=tf.int64
    )
    feat_dict["timestamp"] = DenseFeat(
        name="timestamp",
        dtype=tf.float32,
        scaler=MinMaxScaler()
    )
    feat_dict["age"] = DenseFeat(
        name="age",
        dtype=tf.float32,
        scaler=MinMaxScaler()
    )
    feat_dict["genres"] = MultiValCsvFeat(
        name="genres",
        tags=domains["genres"],
        dtype=tf.string
    )
    # fmt: on

    feat_dict.initialize(df_data)
    return feat_dict
