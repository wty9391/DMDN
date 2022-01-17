from typing import List, Dict
import numpy as np
import sys
import os
import codecs
import json
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.internal import special_math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import feature_column
from tensorflow.python.keras.utils import layer_utils

import util

EPSILON = tf.keras.backend.epsilon() * 2
GAUSSIAN = "gaussian"
LOG_GAUSSIAN = "log_gaussian"
LOGISTIC = "logistic"
GUMBEL = "gumbel"
GAMMA = "gamma"

class Deep_Mixture_Density_Model(Model):
    def __init__(self, training_args, model_args, data_args,
                 feature_col: List[str],
                 numerical_col: List[str],
                 pay_price_col: str,
                 bid_price_col: str,
                 categorical_vocabulary: Dict[str, List[str]],
                 true_pdf: List[float],
                 logger,
                 tb_writer):
        """
        :param feature_col: all feature column name, not include bid_price_col, pay_price_col
        :param numerical_col: numerical feature column name, the rest are categorical features
        :param categorical_vocabulary: vocabulary of categorical features
        """
        super(Deep_Mixture_Density_Model, self).__init__()
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.feature_col = feature_col
        self.numerical_col = numerical_col
        self.pay_price_col = pay_price_col
        self.bid_price_col = bid_price_col
        self.categorical_vocabulary = categorical_vocabulary
        self.true_pdf = tf.constant(true_pdf)
        self.logger = logger
        self.tb_writer = tb_writer

        assert model_args.K > 0
        assert model_args.family in [GAUSSIAN, LOG_GAUSSIAN, LOGISTIC, GUMBEL, GAMMA]

        self.uncompiled_model = self.get_uncompiled_model()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.anlp_tracker = keras.metrics.Mean(name="anlp")
        self.mse_tracker = keras.metrics.Mean(name="mse")
        self.my_tracker = MyMetric(name="kl_wd", true_pdf=self.true_pdf, tb_writer=tb_writer, training_args=training_args)

        self.uncompiled_model.summary()

        with tb_writer.as_default():
            tf.summary.scalar('model/trainable_variables',
                              layer_utils.count_params(self.uncompiled_model.trainable_weights),
                              0)


    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker, self.anlp_tracker, self.mse_tracker, self.my_tracker]


    def __get_layer_embedding(self):
        # not compatible with tf.keras.Model
        # all_embeddings = []
        # for feature_name in self.feature_col:
        #     column = None
        #     if feature_name in self.numerical_col:
        #         # numerical feature
        #         column = feature_column.numeric_column(feature_name)
        #     else:
        #         # categorical feature
        #         vocabulary_list = self.categorical_vocabulary[feature_name]
        #         dimension = int(min(np.ceil(len(vocabulary_list) / 2), self.model_args.embedding_dimension))
        #         categorical_column = feature_column.categorical_column_with_vocabulary_list(key=feature_name,
        #                                                                                     vocabulary_list=vocabulary_list)
        #         column = feature_column.embedding_column(categorical_column, dimension=dimension)
        #     all_embeddings.append(column)
        #
        # return tf.keras.layers.DenseFeatures(all_embeddings)

        inputs = []
        embeddings = []

        for feature_name in self.feature_col:
            column = keras.Input(shape=(1,), name=feature_name)
            embedding = None
            if feature_name in self.numerical_col:
                embedding = layers.Reshape(target_shape=(1,1,))(column)
            else:
                vocabulary_list = self.categorical_vocabulary[feature_name]
                dimension = int(min(np.ceil(len(vocabulary_list) / 2), self.model_args.embedding_dimension))
                embedding = layers.Embedding(len(vocabulary_list), dimension)(column)

            inputs.append(column)
            embeddings.append(embedding)

        # embedding output 3D tensor with shape: (batch_size, input_length, output_dim).
        embdedding_layer = layers.concatenate(embeddings)

        return inputs, layers.Reshape(target_shape=(-1,))(embdedding_layer)


    def get_uncompiled_model(self):
        inputs, layer_embedding = self.__get_layer_embedding()
        inputs.append(keras.Input(shape=(1,), name=self.bid_price_col))
        inputs.append(keras.Input(shape=(1,), name=self.pay_price_col))

        hidden1 = layers.Dense(self.model_args.hidden_dimension,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden1")(layer_embedding)
        hidden1_1 = layers.Dense(self.model_args.hidden_dimension,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden1_1")(hidden1)

        hidden2 = layers.Dense(self.model_args.hidden_dimension/2,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden2")(hidden1_1)
        hidden2_1 = layers.Dense(self.model_args.hidden_dimension/2,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden2_1")(hidden2)

        hidden3 = layers.Dense(self.model_args.hidden_dimension/4,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden3")(hidden2_1)

        hidden3_1 = layers.Dense(self.model_args.hidden_dimension/4,
                               activation='selu',
                               kernel_initializer='lecun_normal',
                               name="hidden_final")(hidden3)

        hidden3_2 = layers.BatchNormalization()(hidden3_1)

        # hidden_final = layers.Dropout(self.model_args.drop_out)(hidden3_2)
        hidden_final = hidden3_2


        init_para_1_min = 0
        init_para_1_max = 1
        init_para_2_min = 0
        init_para_2_max = 1
        one = 1.0 # for numerical stable

        if self.model_args.family in [GAUSSIAN, LOGISTIC]:
            init_para_1_min = 1/self.data_args.price_scale
            init_para_1_max = self.data_args.max_price/self.data_args.price_scale
            init_para_2_min = self.data_args.max_price/self.data_args.price_scale/self.model_args.K/4
            init_para_2_max = self.data_args.max_price/self.data_args.price_scale/self.model_args.K/2
            if self.data_args.dataset == "ipinyou":
                one = 0.4

            if self.data_args.dataset == "criteo":
                init_para_1_max = self.data_args.max_price / self.data_args.price_scale / 10

        elif self.model_args.family == LOG_GAUSSIAN:
            init_para_1_min = 0
            init_para_1_max = tf.cast(tf.math.log(self.data_args.max_price/self.data_args.price_scale), tf.float32)
            init_para_2_min = 0
            # init_para_2_max = tf.cast(tf.math.log(self.data_args.max_price/self.data_args.price_scale)/self.model_args.K, tf.float32)
            init_para_2_max = 1 / self.data_args.price_scale
            one = 0.1
        elif self.model_args.family in [GUMBEL]:
            init_para_1_min = 1/self.data_args.price_scale
            init_para_1_max = self.data_args.max_price/self.data_args.price_scale/1.5
            init_para_2_min = 1/self.data_args.price_scale
            init_para_2_max = self.data_args.max_price/self.data_args.price_scale/self.model_args.K/3
        elif self.model_args.family == GAMMA:
            init_para_1_min = 1/self.data_args.price_scale
            init_para_1_max = self.data_args.max_price/self.data_args.price_scale
            init_para_2_min = 1/self.data_args.price_scale
            init_para_2_max = self.data_args.max_price/self.data_args.price_scale/self.model_args.K/self.model_args.K
        else:
            pass

        out_dist_paras_1 = layers.Dense(self.model_args.K,
                                        activation='softplus',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer=tf.keras.initializers.RandomUniform(minval=init_para_1_min,
                                                                                             maxval=init_para_1_max),
                                        name="dist_paras_1")(hidden_final)
        out_dist_paras_2 = layers.Dense(self.model_args.K,
                                        activation='softplus',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                        bias_initializer=tf.keras.initializers.RandomUniform(minval=init_para_2_min,
                                                                                             maxval=init_para_2_max),
                                        name="dist_paras_2")(hidden_final)
        mixing_coefficients = layers.Dense(self.model_args.K, activation='softmax', name="mixing_coefficients")(hidden_final)

        # numerical stable
        constant_one = tf.constant([one / self.data_args.price_scale])
        lambda_add_one = tf.keras.layers.Lambda(lambda x: x + constant_one)

        # numerical stable
        constant_point_one = tf.constant([one/10/self.model_args.K / self.data_args.price_scale])
        lambda_add_point_one = tf.keras.layers.Lambda(lambda x: x + constant_point_one)

        # out_mixing_coefficients = tf.keras.layers.Dropout(self.model_args.drop_out, input_shape=(-1, self.model_args.K))(mixing_coefficients)
        out_mixing_coefficients = lambda_add_point_one(mixing_coefficients)

        # concat_layer = layers.concatenate([out_mixing_coefficients, out_dist_paras_1, out_dist_paras_2])
        mixing_layer = None
        if self.model_args.family == GAUSSIAN:
            # mixing_layer = tfp.layers.MixtureSameFamily(self.model_args.K, tfp.layers.IndependentNormal(1))(concat_layer)
            # mixing_layer = tfd.MixtureSameFamily(
            #     mixture_distribution=tfd.Categorical(probs=out_mixing_coefficients),
            #     components_distribution=tfd.Independent(tfd.Normal(loc=out_dist_paras_1, scale=out_dist_paras_2)),
            #     validate_args=False)

            mixing_layer = tfp.layers.DistributionLambda(
                make_distribution_fn=Deep_Mixture_Density_Model.__gaussian_mixture_layer)\
                ((out_mixing_coefficients,
                  lambda_add_one(out_dist_paras_1),
                  lambda_add_one(out_dist_paras_2),))
        elif self.model_args.family == LOG_GAUSSIAN:
            mixing_layer = tfp.layers.DistributionLambda(
                make_distribution_fn=Deep_Mixture_Density_Model.__log_gaussian_mixture_layer)\
                ((out_mixing_coefficients,
                  lambda_add_one(out_dist_paras_1),
                  lambda_add_one(out_dist_paras_2),))
        elif self.model_args.family == LOGISTIC:
            mixing_layer = tfp.layers.DistributionLambda(
                make_distribution_fn=Deep_Mixture_Density_Model.__logistic_mixture_layer)\
                ((out_mixing_coefficients,
                  lambda_add_one(out_dist_paras_1),
                  lambda_add_one(out_dist_paras_2),))
        elif self.model_args.family == GUMBEL:
            mixing_layer = tfp.layers.DistributionLambda(
                make_distribution_fn=Deep_Mixture_Density_Model.__gumbel_mixture_layer)\
                ((out_mixing_coefficients,
                  lambda_add_one(out_dist_paras_1),
                  lambda_add_one(out_dist_paras_2),))
        elif self.model_args.family == GAMMA:
            mixing_layer = tfp.layers.DistributionLambda(
                make_distribution_fn=Deep_Mixture_Density_Model.__gamma_mixture_layer)\
                ((out_mixing_coefficients,
                  lambda_add_one(out_dist_paras_1),
                  lambda_add_one(out_dist_paras_2),))

        else:
            pass

        model = keras.Model(
            inputs=inputs,
            # outputs=[out_dist_paras_1, out_dist_paras_2, out_mixing_coefficients],
            outputs=mixing_layer,
        )

        return model

    @staticmethod
    def __gaussian_mixture_layer(inputs):
        mix, mu, sigma = inputs

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix),
            components_distribution=tfd.Normal(loc=mu, scale=sigma),
            validate_args=False)

    @staticmethod
    def __log_gaussian_mixture_layer(inputs):
        mix, mu, sigma = inputs

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix),
            components_distribution=tfd.LogNormal(loc=mu, scale=sigma),
            validate_args=False)

    @staticmethod
    def __logistic_mixture_layer(inputs):
        mix, mu, sigma = inputs

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix),
            components_distribution=tfd.Logistic(loc=mu, scale=sigma),
            validate_args=False)

    @staticmethod
    def __gumbel_mixture_layer(inputs):
        mix, mu, sigma = inputs

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix),
            components_distribution=tfd.Gumbel(loc=mu, scale=sigma),
            validate_args=False)

    @staticmethod
    def __gamma_mixture_layer(inputs):
        mix, mu, sigma = inputs

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix),
            components_distribution=tfd.Gamma(concentration=mu, rate=sigma),
            validate_args=False)

    def log_loss(self, inputs, labels, bids, wins, normalize=False, verbose=False):
        distribution: tfp.distributions.Distribution = self.uncompiled_model(inputs)
        labels = tf.cast(labels, tf.float32)
        labels = tf.math.add(labels, tf.keras.backend.epsilon()) # in case of log(0) in log-normal distribution
        wins = tf.cast(wins, tf.float32)
        win_rate = tf.reduce_sum(wins) / tf.cast(tf.shape(labels)[0], tf.float32)
        log_likelihood = None
        one = tf.ones_like(labels, dtype=tf.float32) / self.data_args.price_scale

        if normalize:
            pdf, z_list = self.pdf_for_z_list(inputs=inputs)
            pdf_sum = tf.reduce_sum(pdf, 0)

            if self.model_args.first_price:
                log_likelihood_for_win = tf.math.log(tf.math.divide(tf.clip_by_value(distribution.cdf(bids-one), 0.0, 1.0), pdf_sum) + tf.keras.backend.epsilon())
                log_likelihood_for_lose = tf.math.log(tf.math.divide(tf.clip_by_value(1 - distribution.cdf(bids-one), 0.0, 1.0), pdf_sum) + tf.keras.backend.epsilon())
                log_likelihood = tf.math.multiply(wins, log_likelihood_for_win) + tf.math.multiply(1-wins, log_likelihood_for_lose)
            else:
                log_likelihood_for_win = tf.math.log(tf.math.divide(distribution.prob(labels), pdf_sum) + tf.keras.backend.epsilon())
                log_likelihood_for_lose = tf.math.log(tf.math.divide(tf.clip_by_value(1 - distribution.cdf(bids-one), 0.0, 1.0), pdf_sum) + tf.keras.backend.epsilon())
                # log_likelihood_cdf = tf.math.log(tf.math.divide(1 - distribution.cdf(bids) + EPSILON, pdf_sum))
                log_likelihood = tf.math.multiply(wins, log_likelihood_for_win) + tf.math.multiply(1-wins, log_likelihood_for_lose)
        else:
            if self.model_args.first_price:
                log_likelihood_for_win = distribution.log_cdf(bids-one)
                log_likelihood_for_lose = tf.math.log(tf.clip_by_value(1 - distribution.cdf(bids-one), 0.0, 1.0) + tf.keras.backend.epsilon())
                log_likelihood = tf.math.multiply(wins, log_likelihood_for_win) + tf.math.multiply(1 - wins,
                                                                                                   log_likelihood_for_lose)
            else:
                log_likelihood_for_win = distribution.log_prob(labels)
                log_likelihood_for_lose = tf.math.log(tf.clip_by_value(1 - distribution.cdf(bids-one), 0.0, 1.0) + tf.keras.backend.epsilon())
                # log_likelihood_cdf = tf.math.log(1 - distribution.cdf(bids) + EPSILON)
                log_likelihood = tf.math.multiply(wins, log_likelihood_for_win) + tf.math.multiply(1-wins, log_likelihood_for_lose)

        if verbose:
            index = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(labels)[0], dtype=tf.int32)
            normal_dist, cat_dist, *_ = distribution.submodules
            tf.print()
            para1 = normal_dist.concentration[index] if self.model_args.family == GAMMA else normal_dist.loc[index]
            para2 = normal_dist.rate[index] if self.model_args.family == GAMMA else normal_dist.scale[index]

            tf.print(tf.math.round(para1),
                     tf.math.round(para2),
                     tf.math.round(cat_dist.probs[index]*100.0),
                     labels[index],
                     win_rate,
                     output_stream=sys.stdout,
                     summarize=-1,
                     sep=',')

        return -tf.reduce_mean(log_likelihood)

    def anlp(self, inputs, labels):
        """
        :param inputs:
        :param labels:
        :return:
        """
        distribution: tfp.distributions.Distribution = self.uncompiled_model(inputs)
        labels = tf.cast(labels, tf.float32)
        labels = tf.math.add(labels, tf.keras.backend.epsilon())

        pdf, z_list = self.pdf_for_z_list(inputs=inputs)
        pdf_sum = tf.reduce_sum(pdf, 0)
        # pdf normalization
        pdf = tf.math.divide(distribution.prob(labels), pdf_sum) + tf.keras.backend.epsilon()

        return -tf.reduce_mean(tf.math.log(pdf))


    def mse_loss(self, inputs, labels):
        labels = tf.cast(labels, tf.float32)

        # shape: (zlist.shape[0], batch_size)
        pdf, z_list = self.pdf_for_z_list(inputs=inputs)
        # pdf normalize
        pdf_sum = tf.reduce_sum(pdf, 0)
        pdf = tf.math.divide(pdf, pdf_sum)
        predict = tf.matmul(tf.transpose(pdf), z_list)

        return tf.reduce_mean(tf.math.squared_difference(tf.reshape(predict, [-1]), labels))

    def pdf_for_z_list(self, inputs):
        distribution: tfp.distributions.Distribution = self.uncompiled_model(inputs)

        # z_list is [0, 1, ... , max_price]
        z_list = tf.constant([(i+tf.keras.backend.epsilon()) / self.data_args.price_scale for i in range(int(self.data_args.max_price)+1)],
                             dtype=tf.float32)  # in case of log(0) in log-normal distribution
        z_list = tf.reshape(z_list, [-1, 1])
        # shape: (zlist.shape[0], batch_size)
        pdf = distribution.prob(z_list)
        return pdf, z_list

    def pdf_sum(self, inputs):
        pdf, z_list = self.pdf_for_z_list(inputs=inputs)
        pdf_sum = tf.reduce_sum(pdf, 0) # pdf normalize
        pdf = tf.math.divide(pdf, pdf_sum)
        pdf = tf.reduce_sum(pdf, 1)

        return pdf

    @tf.function
    def train_step(self, data):
        x, z = data
        with tf.GradientTape() as tape:
            loss = self.log_loss(inputs=x,
                                 labels=z["pay_price"],
                                 bids=z["bid_price"],
                                 wins=z["win"],
                                 normalize=self.training_args.nomalize_loss,
                                 verbose=self.training_args.debug)

        g = tape.gradient(loss, self.trainable_variables)
        g = [(tf.clip_by_value(grad, -0.5, 0.5)) for grad in g]
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss, sample_weight=tf.shape(z["win"])[0])

        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        x, z = data
        loss = self.log_loss(inputs=x,
                             labels=z["pay_price"],
                             bids=z["bid_price"],
                             wins=z["win"],
                             normalize=self.training_args.nomalize_loss,
                             verbose=False)
        # anlp = self.log_loss(inputs=x,
        #                      labels=z["pay_price"],
        #                      bids=z["bid_price"],
        #                      wins=tf.ones_like(z["win"]),
        #                      normalize=True,
        #                      verbose=False)
        anlp = self.anlp(inputs=x,
                         labels=z["pay_price"])
        mse = self.mse_loss(inputs=x,
                            labels=z["pay_price"])
        pdf = self.pdf_sum(inputs=x)

        self.anlp_tracker.update_state(anlp, sample_weight=tf.shape(z["win"])[0])
        self.loss_tracker.update_state(loss, sample_weight=tf.shape(z["win"])[0])
        self.mse_tracker.update_state(mse, sample_weight=tf.shape(z["win"])[0])
        self.my_tracker.update_state(pdf_sum=pdf, count=tf.shape(z["win"])[0])

        return {"loss": self.loss_tracker.result(),
                "anlp": self.anlp_tracker.result(),
                "mse": self.mse_tracker.result(),
                "kl": self.my_tracker.result()[0],
                "wd": self.my_tracker.result()[1]}

class MyMetric(tf.keras.metrics.Metric):
  def __init__(self, true_pdf: tf.Tensor, tb_writer=None, name='kl_divergence', training_args=None, **kwargs):
    super(MyMetric, self).__init__(name=name, **kwargs)
    self.length = tf.shape(true_pdf)[0]
    self.true_pdf = true_pdf
    self.pdf_sum = self.add_weight(shape=(self.length,), name='pdf_sum', initializer='zeros')
    self.pdf_predict = self.add_weight(shape=(self.length,), name='pdf_predict', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
    self.tb_writer = tb_writer
    self.training_args = training_args
    self.epoch = 0

  def update_state(self, pdf_sum, count):

    count = tf.cast(count, tf.float32)
    pdf_sum = tf.cast(pdf_sum, tf.float32)
    self.count.assign_add(count)
    self.pdf_sum.assign_add(pdf_sum)
    self.pdf_predict.assign(tf.divide(self.pdf_sum, self.count))

  def result(self):
    kl = tf.keras.losses.KLDivergence()
    wd = tf.reduce_sum(tf.math.abs(tf.math.subtract(tf.cast(self.true_pdf, tf.float32),
                                                    tf.cast(self.pdf_predict, tf.float32))))

    return kl(self.true_pdf, self.pdf_predict), wd

  def reset_states(self):
    if tf.math.greater(self.count, 0):
          # render pdf image to tensorboard
        if self.tb_writer is not None:
            f, (ax1) = plt.subplots(1, 1)

            z_list = [i for i in range(self.length.numpy())]
            ax1.plot(z_list, self.true_pdf.numpy(), color='tab:orange', label='truth')
            ax1.plot(z_list, self.pdf_predict.numpy(), color='tab:green', label='predict')
            ax1.set_ylabel("probability")
            ax1.set_xlabel("market price")
            ax1.legend()
            f.set_size_inches(7, 7 / 16 * 9)
            plt.tight_layout()

            with self.tb_writer.as_default():
                tf.summary.image("test pdf", util.plot_to_image(f), step=self.epoch)

            # save pdf file
            pdf_path = os.path.join(self.training_args.output_dir, "pdf", "pdf_{}.json".format(self.epoch))
            Path(os.path.dirname(pdf_path)).mkdir(parents=True, exist_ok=True)

            obj = {
                "z_list": z_list,
                "pdf": self.pdf_predict.numpy().tolist(),
                "pdf_true": self.true_pdf.numpy().tolist()
            }

            with codecs.open(pdf_path, 'wb+', 'utf-8') as f:
                f.write(json.dumps(obj))


        # reset state
        self.pdf_sum.assign(tf.zeros_like(self.pdf_sum, dtype=tf.float32))
        self.pdf_predict.assign(tf.zeros_like(self.pdf_predict, dtype=tf.float32))
        self.count.assign(0)
        self.epoch += 1
