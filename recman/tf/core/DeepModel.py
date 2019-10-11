import logging
from abc import ABC, ABCMeta, abstractmethod
from time import time

import numpy as np
import tensorflow as tf
import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle

from .inputs import DataInputs, FeatureDictionary


log = logging.getLogger(__name__)


class DeepModel(BaseEstimator, TransformerMixin, ABC):

    __metaclass__ = ABCMeta

    def __init__(
            self,
            feat_dict: FeatureDictionary,
            hparams: dict,
            metrics,
            epoch,
            batch_size=64,
            random_seed=2019,
            task="classification",
    ):
        assert task in [
            "classification",
            "regression",
        ], "target can be either 'classification' for classification task or 'regression' for regression task"

        self.task = task
        self.feat_dict = feat_dict
        self.hparams = hparams
        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.metrics = metrics
        self.variables = dict()

    def predict(self, X, training=False):
        dummy_y = np.array([1] * len(X))
        y_pred = None

        total_batch = len(dummy_y) // self.batch_size + 1
        for batch_index in range(total_batch):
            x_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)
            num_batch = len(y_batch)

            inputs = DataInputs()
            inputs.load(self.feat_dict, x_batch, y_batch)

            batch_out = self._out(inputs, training=training)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            if batch_index % 50 == 0:
                log.info(
                    f"Predict: {(batch_index + 1)}/{total_batch} has been completed"
                )
        log.info(f"Predict: {total_batch}/{total_batch} has been completed")

        return y_pred

    def evaluate(self, X, y, training=False):
        pred = self.predict(X, training)
        return [metric(y, pred) for metric in self.metrics]

    @staticmethod
    def get_batch(X, y, batch_size, index):
        start = index * batch_size
        end = start + batch_size
        end = end if end < len(y) else len(y)
        return X[start:end], y[start:end]

    def load(self):
        ckpt = tf.train.Checkpoint(**self.variables)
        status = ckpt.restore(tf.train.latest_checkpoint("."))
        status.assert_consumed()

    @abstractmethod
    def fit_on_batch(self, X, y):
        pass

    def _eval_at_epoch(
            self, X_train, y_train, X_valid=None, y_valid=None, start_time=time(), epoch=0
    ):
        has_valid = X_valid is not None and y_valid is not None
        train_res = self.evaluate(X_train, y_train, training=True)

        if has_valid:
            valid_res = self.evaluate(X_valid, y_valid, training=True)
            print(
                "[%d] train-result=%s, valid-result=%s [%.1f s]"
                % (
                    epoch,
                    str([(f, round(r, 4)) for f, r in zip(self.metrics, train_res)]),
                    str([(f, round(r, 4)) for f, r in zip(self.metrics, valid_res)]),
                    time() - start_time,
                )
            )
            return train_res, valid_res
        else:
            print(
                "[%d] train-result=%s [%.1f s]"
                % (
                    epoch,
                    str([(f, round(r, 4)) for f, r in zip(self.metrics, train_res)]),
                    time() - start_time,
                )
            )
            return train_res, None

    @abstractmethod
    def _out(self, inputs, training=True):
        pass

    @abstractmethod
    def _loss(self, inputs):
        pass

    def fit(
            self,
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            random_seed_for_mini_batch=True,
            show_progress=False,
            tb_logger=None,
            epoch_callback=None,
    ):
        assert X_train is not None or y_train is not None

        if tb_logger is not None:
            tb_logger.configure_hparams(self.hparams, self.metrics)

        with tqdm.tqdm(
                desc="fit", total=self.epoch, disable=not show_progress
        ) as progress:
            if tb_logger is not None:
                tb_logger.trace_on(epoch=0, graph=True, profiler=False)
            eval_results = self._eval_at_epoch(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                start_time=time(),
            )
            if tb_logger is not None:
                tb_logger.trace_off(epoch=0)
                tb_logger.log_params(
                    epoch=0,
                    eval_results=eval_results,
                    metrics=self.metrics,
                    variables=self.variables,
                )

            best_score = None
            for epoch in range(1, self.epoch + 1):
                start_time = time()
                if random_seed_for_mini_batch:
                    seed = np.random.randint(1, 2019)
                else:
                    seed = self.random_seed
                X_train = shuffle(X_train, random_state=seed)
                y_train = shuffle(y_train, random_state=seed)
                total_batch = len(y_train) // self.batch_size + 1

                if tb_logger is not None:
                    tb_logger.trace_on(epoch, graph=False, profiler=True)

                for i in range(total_batch):
                    Xi_batch, y_batch = self.get_batch(
                        X_train, y_train, self.batch_size, i
                    )

                    self.fit_on_batch(Xi_batch, y_batch)

                    if i % 50 == 0:
                        log.info(f"Fit: {(i + 1)}/{total_batch} has been completed")
                log.info(f"Fit: {total_batch}/{total_batch} has been completed")

                if tb_logger is not None:
                    tb_logger.trace_off(epoch)

                eval_results = self._eval_at_epoch(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    start_time=time(),
                )
                if tb_logger is not None:
                    tb_logger.log_params(
                        epoch=epoch,
                        eval_results=eval_results,
                        metrics=self.metrics,
                        variables=self.variables,
                    )

                if epoch_callback:
                    epoch_callback(self.variables, eval_results)

                progress.update(1)
