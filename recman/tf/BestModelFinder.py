import tensorflow as tf
# import dill as pickle


class BestModelFinder:
    def __init__(self):
        self.best_score = None
        self.hp_val = None

    def set_hparams(self, hp_val):
        self.hp_val = hp_val

    def __call__(self, variables, eval_results):
        eval_results = list(filter(lambda r: r, eval_results))
        score = eval_results[-1][0]
        if self.best_score is None or score < self.best_score:
            print("Best model found!")
            self.best_score = score

            ckpt = tf.train.Checkpoint(**variables)
            ckpt.save(f"./ckpt_model")

            pickle.dump(
                self.hp_val, open("./hparams", "wb"), protocol=pickle.HIGHEST_PROTOCOL
            )
