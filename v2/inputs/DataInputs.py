import tensorflow as tf

class DataInputs(dict):
    def load(self, feat_dict, X, y):
        for feat in feat_dict.values():
            self[feat.name] = tf.convert_to_tensor(
                feat(X[feat.name]), name=f"{feat.name}_input"
            )
        self["y"] = tf.convert_to_tensor(y, name="y")
