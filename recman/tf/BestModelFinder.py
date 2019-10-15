import tensorflow as tf
import dill as pickle
import logging

log = logging.getLogger(__name__)


class BestModelFinder:
    def __init__(self):
        self._best_score = None
        self._best_hp_val = None
        self._best_eval_results = None

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_eval_results(self):
        return self._best_eval_results

    @property
    def best_hp_val(self):
        return self._best_hp_val

    def __call__(self, **kwargs):
        variables = kwargs["variables"]
        eval_results = kwargs["eval_results"]
        hp_val = kwargs["hp_val"]
        feat_dict = kwargs["feat_dict"]
        df_all = kwargs["df_all"]

        assert (
            variables is not None
            and eval_results is not None
            and hp_val is not None
            and feat_dict is not None
        )

        # filter the empty ones
        eval_results = list(filter(lambda r: r, eval_results))
        score = eval_results[-1][0]

        if self._best_score is None or score < self._best_score:
            log.info("A better model is found!")
            log.info(eval_results)

            self._best_score = score
            self._best_hp_val = eval_results
            self._best_eval_results = eval_results

            ckpt = tf.train.Checkpoint(**variables)
            ckpt.save(f"./ckpt_model")

            with open("./hparams", "wb") as writer:
                pickle.dump(hp_val, writer, protocol=pickle.HIGHEST_PROTOCOL)

            with open("./feat_dict", "wb") as writer:
                pickle.dump(feat_dict, writer, protocol=pickle.HIGHEST_PROTOCOL)

            with open("./df_all", "wb") as writer:
                pickle.dump(df_all, writer, protocol=pickle.HIGHEST_PROTOCOL)
