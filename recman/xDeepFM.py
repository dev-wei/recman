import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

from .DeepModel import DeepModel
from .layers import (
    CIN,
    DNN,
    DNNCombiner,
    FeatEmbeddingLayer,
    LinearCombiner,
    LinearLayer,
    OutputLayer,
)
from .utils import (
    count_parameters,
    create_feat_inputs,
    create_loss,
    create_optimizer,
    initialize_variables,
    tensor_board,
)


class xDeepFM(DeepModel):
    """
    xDeepFM
    
    https://arxiv.org/pdf/1803.05170.pdf
    """

    def __init__(
        self,
        feat_dict: dict,
        embedding_size=8,
        embedding_l2_reg=0.00001,
        linear_l2_reg=0.00001,
        deep_hidden_units=(32, 32),
        deep_dropout=(0.6, 0.6, 0.6),  # good for range (0.6-0.9)
        deep_activation=tf.nn.relu,
        deep_l2_reg=0.0,
        cin_hidden_units=(128, 128),
        cin_dropout=(1, 1, 1),  # good for range (0.6-0.9)
        cin_split_half=True,
        cin_activation=tf.nn.relu,
        l2_reg=0.1,
        epoch=10,
        batch_size=256,
        learning_rate=0.001,
        optimizer="adam",
        random_seed=2019,
        use_deep=True,
        use_cin=True,
        loss_type="logloss",
        eval_metric=(roc_auc_score, log_loss),
        what_means_greater=None,
        use_interactive_session=True,
        log_dir="./logs",
    ):
        DeepModel.__init__(
            self,
            feat_dict,
            epoch,
            batch_size,
            random_seed,
            eval_metric,
            what_means_greater,
            use_interactive_session,
        )

        assert use_cin or use_deep
        assert loss_type in [
            "logloss",
            "mse",
        ], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.loss_type = loss_type

        self.embedding_size = embedding_size
        self.embedding_l2_reg = embedding_l2_reg
        self.linear_l2_reg = linear_l2_reg
        self.cin_hidden_units = cin_hidden_units
        self.cin_split_half = cin_split_half
        self.cin_activation = cin_activation
        self.cin_dropout = cin_dropout
        self.deep_hidden_units = deep_hidden_units
        self.deep_dropout = deep_dropout
        self.deep_activation = deep_activation
        self.deep_l2_reg = deep_l2_reg
        self.l2_reg = l2_reg

        self.use_cin = use_cin
        self.use_deep = use_deep
        self.log_dir = log_dir
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope("FeatureInputs"):
                # inputs
                self.inputs.update(create_feat_inputs(self.feat_dict))

                self.hp_deep_dropout = tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="hp_deep_dropout"
                )
                self.hp_cin_dropout = tf.compat.v1.placeholder(
                    tf.float32, shape=[None], name="hp_cin_dropout"
                )
                self.label = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 1], name="label"
                )
                self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")

            with tf.name_scope("Embeddings"):
                feat_embeds = FeatEmbeddingLayer(
                    self.feat_dict, self.embedding_size, self.embedding_l2_reg
                )
                self.feat_embeds_dict, self.feat_bias_dict = feat_embeds(
                    self.inputs, self.train_phase
                )
                self.weights.update(feat_embeds.weights)
                self.feat_embeds, self.feat_bias = (
                    tf.concat(list(self.feat_embeds_dict.values()), axis=1),
                    tf.concat(list(self.feat_bias_dict.values()), axis=1),
                )

            with tf.name_scope("Linear"):
                linear_combiner = LinearCombiner(self.feat_dict)
                linear_inputs = linear_combiner(self.inputs)

                linear = LinearLayer(self.feat_dict, self.linear_l2_reg)
                self.linear_logit = linear(linear_inputs, self.train_phase)
                self.weights.update(linear.weights)

            if self.use_cin:
                with tf.name_scope("CompressedInteractionNetwork"):
                    cin = CIN(
                        self.cin_hidden_units,
                        self.cin_split_half,
                        self.cin_activation,
                        dropout=self.cin_dropout,
                    )
                    self.cin_logit = cin(self.feat_embeds)
                    self.weights.update(cin.weights)

            if self.use_deep:
                with tf.name_scope("DeepNeuralNetwork"):
                    dnn_combiner = DNNCombiner()
                    dnn_input = dnn_combiner(
                        [self.feat_embeds] + self.inputs.dense_inputs
                    )

                    dnn = DNN(
                        self.deep_hidden_units,
                        self.deep_dropout,
                        self.deep_activation,
                        self.deep_l2_reg,
                    )
                    self.dnn_logit = dnn(dnn_input)
                    self.weights.update(dnn.weights)

            with tf.name_scope("exDeepFM"):
                if self.use_cin and self.use_deep:
                    self.final_logit = tf.add_n(
                        [self.linear_logit, self.cin_logit, self.dnn_logit]
                    )
                elif self.use_cin:
                    self.final_logit = tf.add(self.linear_logit, self.cin_logit)
                elif self.use_deep:
                    self.final_logit = tf.add(self.linear_logit, self.dnn_logit)

            with tf.name_scope("Output"):
                output = OutputLayer(use_bias=False)
                self.out = output(self.final_logit)
                self.weights.update(output.weights)

            with tf.name_scope("Loss"):
                self.loss = create_loss(self.label, self.out, self.loss_type)
                tf.compat.v1.summary.scalar("loss", self.loss)

            with tf.name_scope("L2"):
                l2_loss = feat_embeds.l2()
                tf.compat.v1.summary.scalar("feat_embeds_l2", l2_loss)
                self.loss += l2_loss

                l2_loss = linear.l2()
                tf.compat.v1.summary.scalar("linear_l2", l2_loss)
                self.loss += l2_loss

                if self.use_cin:
                    l2_loss = cin.l2()
                    tf.compat.v1.summary.scalar("cin_l2", l2_loss)
                    self.loss += l2_loss

                if self.use_deep:
                    l2_loss = dnn.l2()
                    tf.compat.v1.summary.scalar("dnn_l2", l2_loss)
                    self.loss += l2_loss

                tf.compat.v1.summary.scalar("loss_after_l2", self.loss)

            with tf.name_scope("Optimizer"):
                self.optimizer = create_optimizer(
                    self.optimizer_type, self.learning_rate, self.loss
                )

            self.sess = initialize_variables(self.interactive_session)
            self.tb_ops, self.tb_writer = tensor_board(self.graph, log_dir=self.log_dir)

            count_parameters()

    @property
    def tb(self):
        return self.tb_ops, self.tb_writer

    @property
    def session(self):
        return self.sess

    @property
    def output(self):
        return self.out

    def create_feed_dict(self, X, y, training=True):
        feed_dict = dict()
        for feat in self.feat_dict.values():
            feed_dict[self.inputs[feat]] = feat(X[feat.name])

        if training:
            feed_dict[self.hp_deep_dropout] = self.deep_dropout
            feed_dict[self.hp_cin_dropout] = self.cin_dropout
        else:
            feed_dict[self.hp_deep_dropout] = [1] * len(self.deep_dropout)
            feed_dict[self.hp_cin_dropout] = [1] * len(self.cin_dropout)

        feed_dict[self.label] = y
        feed_dict[self.train_phase] = training
        return feed_dict

    def fit_on_batch(self, X, y):
        loss, opt = self.session.run(
            [self.loss, self.optimizer], feed_dict=self.create_feed_dict(X, y)
        )
