from datetime import datetime
import itertools
import logging
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from ..hparams import BaseHyperParameters


log = logging.getLogger(__name__)


class TensorBoardLogger:
    def __init__(
            self, hp_params: BaseHyperParameters, sess_num, log_dir="./logs", run_name=None
    ):
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.hp_params = hp_params
        self.log_dir = log_dir
        self.run_name = run_name
        self.sess_num = sess_num

        self.trace_writer = tf.summary.create_file_writer(
            f"{self.log_dir}/{self.run_name}/hp_{self.sess_num}"
        )

    def configure_hparams(self, hp_val, metrics):
        log.info(f"Configure hyper-params on session: {self.sess_num}")
        log.info(hp_val)

        with self.trace_writer.as_default():
            hp.hparams_config(
                hparams=[hparam.tf_hparam for hparam in self.hp_params.values()],
                metrics=[
                    hp.Metric(
                        f"{prefix}{eval_func}", display_name=f"{prefix}{eval_func}"
                    )
                    for prefix, eval_func in list(
                        itertools.product(
                            *list([["TRAIN_", "VALID_", "TEST_"], metrics])
                        )
                    )
                ],
            )
            hp.hparams(
                dict(
                    (
                        self.hp_params[param_name].tf_hparam,
                        param_val
                        if not self.hp_params[param_name].advanced_dtype
                        else str(param_val),
                    )
                    for param_name, param_val in hp_val.items()
                )
            )

    def trace_on(self, epoch=-1, graph=False, profiler=True):
        log.info(f"Turn on tracing on session: {self.sess_num} epoch: {epoch}")
        tf.summary.trace_on(graph=graph, profiler=profiler)

    def trace_off(self, epoch=-1):
        log.info(f"Turn off tracing on session: {self.sess_num} epoch: {epoch}")
        with self.trace_writer.as_default():
            tf.summary.trace_export(
                name=f"fit_graph",
                step=epoch,
                profiler_outdir=f"{self.log_dir}/{self.run_name}/hp_{self.sess_num}",
            )

    def log_params(self, epoch, eval_results, metrics, variables):
        with self.trace_writer.as_default():
            for name, value in variables.items():
                tf.summary.histogram(name, value, step=epoch)

            self._log_eval(epoch, eval_results, metrics)

    def _log_eval(self, epoch, eval_results, metrics):
        train_res, valid_res = eval_results

        for f, r in zip(metrics, train_res):
            tf.summary.scalar(f"TRAIN_{f}", r, step=epoch)

        if valid_res is not None:
            for f, r in zip(metrics, valid_res):
                tf.summary.scalar(f"VALID_{f}", r, step=epoch)
