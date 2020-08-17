import tensorflow as tf

from .utils import glorot_normal
from ..inputs import DenseFeat, SparseFeat


class FeatEmbedding:
    """
    Feature Embedding
    """

    def __init__(
        self,
        variables,
        feat,
        embedding_size,
        l2_reg=0.00001,
        use_bias=True,
        prefix="",
        seed=2019,
    ):
        assert not isinstance(feat, DenseFeat)

        self.variables = variables
        self.feat = feat
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.prefix = prefix
        self.seed = seed

    def _upsert_variables(self):
        name = f"{self.prefix}{self.feat.name}_feat_embed"
        if name not in self.variables:
            self.variables[name] = tf.Variable(
                glorot_normal(
                    [self.feat.feat_size, self.embedding_size], seed=self.seed
                ),
                dtype=tf.float32,
                name=name,
            )

        name = f"{self.prefix}{self.feat.name}_feat_bias"
        if name not in self.variables and self.use_bias:
            self.variables[name] = tf.Variable(
                tf.zeros([self.feat.feat_size, 1]), name=name, dtype=tf.float32
            )

    def __call__(self, feat_input):
        with tf.name_scope(f"{self.prefix}{self.feat.name}_{self.__class__.__name__}"):
            self._upsert_variables()

            feat_bias = None
            if isinstance(self.feat, SparseFeat):
                feat_embeds = tf.nn.embedding_lookup(
                    self.variables[f"{self.prefix}{self.feat.name}_feat_embed"],
                    feat_input[:, :1],
                    name=f"{self.prefix}{self.feat.name}_embed_lookup",
                )
                if self.use_bias:
                    feat_bias = tf.nn.embedding_lookup(
                        self.variables[f"{self.prefix}{self.feat.name}_feat_bias"],
                        feat_input[:, :1],
                        name=f"{self.prefix}{self.feat.name}_bias_lookup",
                    )

        return feat_embeds, feat_bias

    def l2(self):
        return tf.multiply(
            self.l2_reg,
            tf.nn.l2_loss(self.variables[f"{self.prefix}{self.feat.name}_feat_embed"]),
            name=f"{self.prefix}{self.feat.name}_l2",
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(feat={self.feat}, "
            f"embedding_size={self.embedding_size}, "
            f"use_bias={self.use_bias})"
        )


class FeatEmbeddingLayer:
    """
    Feature Embeddings, a set contains all input feature embeddings
    """

    def __init__(
        self,
        variables,
        feat_dict,
        embedding_size,
        l2_reg=0.00001,
        use_bias=True,
        prefix="",
        seed=2019,
    ):
        self.variables = variables
        self.feat_dict = feat_dict
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.prefix = prefix
        self.seed = seed

        # TODO: sort the list by putting the Sequence features to the last
        self.feat_embeds = dict(
            (
                feat,
                FeatEmbedding(
                    self.variables,
                    feat,
                    self.embedding_size,
                    self.l2_reg,
                    use_bias=use_bias,
                    prefix=prefix,
                    seed=self.seed,
                ),
            )
            for feat in self.feat_dict.embedding_feats
        )

    def __call__(self, inputs):
        with tf.name_scope(f"{self.prefix}{self.__class__.__name__}"):
            self.feat_embeds_dict, self.feat_bias_dict = dict(), dict()

            for feat in self.feat_embeds:
                feat_embed, feat_bias = self.feat_embeds[feat](inputs[feat.name])
                self.feat_embeds_dict[feat.name] = feat_embed
                if self.use_bias:
                    self.feat_bias_dict[feat.name] = feat_bias

            return (
                tf.concat(
                    list(self.feat_embeds_dict.values()),
                    axis=1,
                    name=f"{self.prefix}feat_embeds",
                ),
                tf.concat(
                    list(self.feat_bias_dict.values()),
                    axis=1,
                    name=f"{self.prefix}feat_bias",
                )
                if self.use_bias
                else None,
            )

    def l2(self):
        return tf.add_n(
            [feat_embed.l2() for feat_embed in self.feat_embeds.values()],
            name=f"{self.prefix}feat_embeds_l2",
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}(feat_embeds={self.feat_embeds})"
