from abc import ABCMeta, abstractmethod
from time import time

import numpy as np
import tensorflow as tf
import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle

from .inputs import FeatureDictionary, FeatureInputs
from .tb import log_scalar


class DeepModel(BaseEstimator, TransformerMixin):

    __metaclass__ = ABCMeta

    def __init__(
        self,
        feat_dict: dict,
        epoch,
        batch_size,
        random_seed,
        eval_metric,
        what_means_greater,
        use_interactive_session,
    ):
        self.feat_dict: FeatureDictionary = feat_dict
        self.inputs: FeatureInputs = FeatureInputs()
        self.weights = dict()

        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.eval_metric = eval_metric
        self.what_means_greater = what_means_greater
        self.train_results, self.valid_results = [], []
        self.interactive_session = use_interactive_session

        tf.compat.v1.set_random_seed(self.random_seed)

    def predict(self, X, training=False):
        dummy_y = np.array([1] * len(X))
        batch_index = 0
        X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)
        y_pred = None

        while len(X_batch):
            num_batch = len(y_batch)

            batch_out = self.session.run(
                self.output,
                feed_dict=self.create_feed_dict(X_batch, y_batch, training=training),
            )
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, X, y, training=False):
        y_hat = self.predict(X, training=training).astype(np.float64)
        return dict(
            (evaluate.__name__, evaluate(y, y_hat)) for evaluate in self.eval_metric
        )

    def get_batch(self, X, y, batch_size, index):
        start = index * batch_size
        end = start + batch_size
        end = end if end < len(y) else len(y)
        return X[start:end], y[start:end].reshape((-1, 1))

    def save(self, file_path):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, file_path)

    def load(self, file_path):
        saver = tf.train.Saver()
        saver.restore(self.session, file_path)

    @property
    @abstractmethod
    def tb(self):
        pass

    @property
    @abstractmethod
    def session(self):
        pass

    @property
    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def create_feed_dict(self, X, y, training=True):
        pass

    @abstractmethod
    def fit_on_batch(self, X, y, training=True):
        pass

    def _collect_feedback(
        self, X_train, y_train, X_valid=None, y_valid=None, start_time=time(), epoch=-1
    ):
        has_valid = X_valid is not None

        train_results = self.evaluate(X_train, y_train, training=True)
        self.train_results.append(train_results)
        if has_valid:
            valid_results = self.evaluate(X_valid, y_valid, training=True)
            self.valid_results.append(valid_results)

            print(
                "[%d] train-result=%s, valid-result=%s [%.1f s]"
                % (
                    epoch + 1,
                    str([(f, round(r, 4)) for f, r in train_results.items()]),
                    str([(f, round(r, 4)) for f, r in valid_results.items()]),
                    time() - start_time,
                )
            )
            return train_results, valid_results
        else:
            print(
                "[%d] train-result=%s [%.1f s]"
                % (
                    epoch + 1,
                    str([(f, round(r, 4)) for f, r in train_results.items()]),
                    time() - start_time,
                )
            )
            return train_results, dict()

    def _log_eval(self, train_results: dict, valid_results: dict = None):
        tf_summaries = []
        for f, r in train_results.items():
            tf_summaries.append(log_scalar(tag=f"Training: {f}", value=r))

        if valid_results is not None:
            for f, r in valid_results.items():
                tf_summaries.append(log_scalar(tag=f"Validating: {f}", value=r))

        return tf_summaries

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        random_seed_for_mini_batch=True,
        early_stopping=False,
        show_progress=False,
        **kwargs,
    ):
        tb_ops, tb_writer = self.tb

        with tqdm.tqdm(
            desc="fit", total=self.epoch, disable=not show_progress
        ) as progress:
            train_results, valid_results = self._collect_feedback(
                X_train, y_train, X_valid, y_valid, start_time=time()
            )
            result = self.session.run(
                tb_ops,
                feed_dict=self.create_feed_dict(
                    X_train[:1], y_train[:1], training=True
                ),
            )
            tb_writer.add_summary(result, 0)
            for tf_sum in self._log_eval(train_results, valid_results):
                tb_writer.add_summary(tf_sum, 0)
            tb_writer.flush()
            for epoch in range(self.epoch):
                t1 = time()

                if random_seed_for_mini_batch:
                    seed = np.random.randint(1, 2019)
                else:
                    seed = self.random_seed
                X_train = shuffle(X_train, random_state=seed)
                y_train = shuffle(y_train, random_state=seed)

                total_batch = len(y_train) // self.batch_size + 1
                for i in range(total_batch):
                    Xi_batch, y_batch = self.get_batch(
                        X_train, y_train, self.batch_size, i
                    )
                    self.fit_on_batch(Xi_batch, y_batch)

                train_results, valid_results = self._collect_feedback(
                    X_train, y_train, X_valid, y_valid, start_time=t1, epoch=epoch
                )

                result = self.session.run(
                    tb_ops,
                    feed_dict=self.create_feed_dict(
                        X_train[:1], y_train[:1], training=True
                    ),
                )
                tb_writer.add_summary(result, epoch + 1)
                for tf_sum in self._log_eval(train_results, valid_results):
                    tb_writer.add_summary(tf_sum, epoch + 1)
                tb_writer.flush()
                progress.update(1)
