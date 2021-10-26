import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TerminateOnNaN(keras.callbacks.Callback):
    def __init__(self, loss_name="loss"):
        super(TerminateOnNaN, self).__init__()
        self.nan_loss = False
        self.loss_name = loss_name

    def on_train_batch_end(self, batch, logs=None):
        if logs[self.loss_name] != logs[self.loss_name]:
            self.model.stop_training = True
            self.nan_loss = True
            print("Stop bcz of nan loss")

class BaseModel(BaseEstimator):
    """
    TODO
    """
    def __init__(self,
        # BaseModel HPs
        num_hidden_layers=2,
        size_hidden_layers=20,
        num_bias_layers=2,
        size_bias_layers=20,
        feature_activation='tanh',
        kernel_regularizer=0.0,
        drop_out=0,
        gamma=1.,
        # Common HPs
        batch_size=200,
        learning_rate=0.001,
        learning_rate_decay_rate=1,
        learning_rate_decay_steps=1000,
        optimizer="Adam",# 'Nadam' 'SGD'
        epoch=10,
        ranking_loss='categorical_crossentropy',
        fair_loss='binary_crossentropy',
        # other variables
        verbose=0,
        validation_size=0.0,
        num_features=0,
        random_seed=42,
        name="BaseModel",
        dataset_name="Compas",
        print_summary=False,
        interpretable=False,
        nan_loss=False
        ):

        # DirectRanker HPs
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.hidden_layers = [size_hidden_layers for i in range(num_hidden_layers)]
        if interpretable:
            self.hidden_layers.append(num_features)
        self.bias_layers = [size_bias_layers for i in range(num_bias_layers)]
        self.feature_activation = feature_activation
        self.kernel_regularizer = kernel_regularizer
        self.drop_out = drop_out
        self.gamma = gamma
        # Common HPs
        self.batch_size = batch_size
        self.ranking_loss = ranking_loss
        self.fair_loss = fair_loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam
        self.optimizer_name = optimizer
        self.epoch = epoch
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.random_seed = random_seed
        self.name = name
        self.dataset_name = dataset_name
        self.print_summary = print_summary
        self.interpretable = interpretable
        self.nan_loss = nan_loss

    def _get_hidden_layer(
        self, 
        input_layer, 
        hidden_layer=[10, 5], 
        drop_out=0, 
        feature_activation="tanh", 
        last_activation="", 
        reg=0, 
        name="",
        print_summary=False
        ):

        nn = tf.keras.layers.Dense(
            units=hidden_layer[0],
            activation=feature_activation,
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            bias_regularizer=tf.keras.regularizers.l2(reg),
            activity_regularizer=tf.keras.regularizers.l2(reg),
            name="nn_{}_0".format(name)
        )(input_layer)

        if drop_out > 0:
            nn = tf.keras.layers.Dropout(drop_out)(nn)

        for i in range(1, len(hidden_layer)):
            nn = tf.keras.layers.Dense(
                units=hidden_layer[i],
                activation=feature_activation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="nn_{}_{}".format(name, i)
            )(nn)

            if drop_out > 0:
                nn = tf.keras.layers.Dropout(drop_out)(nn)

        if last_activation != "":
            nn = tf.keras.layers.Dense(
                units=1,
                activation=last_activation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="nn_out_{}".format(name)
            )(nn)
            
        hidden_part = tf.keras.models.Model(input_layer, nn, name=name)

        if print_summary:
            hidden_part.summary()

        return hidden_part

    def _get_ranking_part(
        self, 
        input_layer,
        num_relevance_classes=1, 
        feature_activation="tanh",
        reg=0,
        use_bias=False,
        name="ranking_part"
        ):

        out = tf.keras.layers.Dense(
            units=num_relevance_classes,
            activation=feature_activation,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            activity_regularizer=tf.keras.regularizers.l2(reg),
            name=name
        )(input_layer)

        return out

    def _build_model(self):
        """
        TODO
        """
        pass

    def fit(self, x, y, s, **fit_params):
        """
        TODO
        x: numpy array of shape [num_instances, num_features]
        y: numpy array of shape [num_instances, num_relevance_classes]
        s: numpy array of shape [num_instances, 1]
        """
        pass
 

    def predict_proba(self, features):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)[0]

        return res

    def predict(self, features, threshold):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)

        return [1 if r > threshold else 0 for r in res[0]]

    def get_representations(self, x):
        return self.feature_part(x)

    def _get_instances(self, x, y, samples):
        """
        :param x:
        :param y:
        :param y_bias:
        :param samples:
        """
        x0 = []
        x1 = []
        y_train = []

        keys, counts = np.unique(y, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        counts = counts[sort_ids]
        for i in range(len(keys) - 1):
                indices0 = np.random.randint(0, counts[i + 1], samples)
                indices1 = np.random.randint(0, counts[i], samples)
                querys0 = np.where(y == keys[i + 1])[0]
                querys1 = np.where(y == keys[i])[0]
                x0.extend(x[querys0][indices0])
                x1.extend(x[querys1][indices1])
                y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y_train = np.array([y_train]).transpose()

        return x0, x1, y_train

    def to_dict(self):
        """
        Return a dictionary representation of the object while dropping the tensorflow stuff.
        Useful to keep track of hyperparameters at the experiment level.
        """
        d = dict(vars(self))

        for key in ['optimizer', 'nan_loss']:
            try:
                d.pop(key)
            except KeyError:
                pass

        return d
