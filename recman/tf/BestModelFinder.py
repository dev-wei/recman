from recman.tf.core import DeepModel
import tensorflow as tf
import dill as pickle
import logging

log = logging.getLogger(__name__)


class BestModelFinder:
    def __init__(self, save_model=False):
        self._best_score = None
        self._best_eval_results = None
        self._best_model = None
        self.save_model = save_model

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_eval_results(self):
        return self._best_eval_results

    @property
    def best_model(self) -> DeepModel:
        return self._model

    def __call__(self, **kwargs):
        assert (
            kwargs["model"] is not None
            and kwargs["model"].hparams is not None
            and kwargs["model"].feat_dict is not None
            and kwargs["model"].variables is not None
            and kwargs["eval_results"] is not None
            and kwargs["df_all"] is not None
        )

        model = kwargs["model"]
        hp_val = model.hparams
        feat_dict = model.feat_dict
        variables = model.variables
        eval_results = kwargs["eval_results"]
        df_all = kwargs["df_all"]

        # filter the empty ones
        eval_results = list(filter(lambda r: r, eval_results))
        score = eval_results[-1][0]

        if self._best_score is None or score < self._best_score:
            log.info("A better model is found!")
            log.info(eval_results)

            self._best_score = score
            self._best_eval_results = eval_results
            self._model = model

            if self.save_model:
                ckpt = tf.train.Checkpoint(**variables)
                ckpt.save(f"./ckpt_model")

                with open("./hparams", "wb") as writer:
                    pickle.dump(hp_val, writer, protocol=pickle.HIGHEST_PROTOCOL)

                with open("./feat_dict", "wb") as writer:
                    pickle.dump(feat_dict, writer, protocol=pickle.HIGHEST_PROTOCOL)

                with open("./df_all", "wb") as writer:
                    pickle.dump(df_all, writer, protocol=pickle.HIGHEST_PROTOCOL)
