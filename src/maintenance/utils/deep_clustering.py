import logging
import os
from datetime import datetime
from time import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.linear_assignment_ import linear_assignment
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

logger = logging.getLogger()

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

_nmi = normalized_mutual_info_score
_ari = adjusted_rand_score

MAX_JOBS = 6


def _acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class DeepClusteringBase:
    def __init__(self, input_shape, autoencoder_ctor, n_clusters,
                 pretrain_epochs, log_dir, tol=1e-3,
                 save_dir='results/temp', max_epochs=100):
        self._autoencoder = autoencoder_ctor(input_shape)
        self._n_clusters = n_clusters
        self._pretrain_epochs = pretrain_epochs
        self._pretrained = False
        self._tol = tol
        self._save_dir = save_dir
        self._max_epochs = max_epochs
        self._log_dir = log_dir

        self._encoder = Model(inputs=self._autoencoder.input, outputs=self._hidden_layer())
        self._model = self.create_model()

    def _hidden_layer(self):
        return self._autoencoder.get_layer(name='embedding').output

    def pretrain(self, x, batch_size=64, epochs=200, optimizer='adam'):
        logger.info('Pretraining...')
        self._autoencoder.compile(optimizer=optimizer, loss='mse')

        STRFTIME = "%Y-%m-%d_%H:%M"
        csv_logger = CSVLogger(os.path.join(self._save_dir, f'pretrain_log_{datetime.now().strftime(STRFTIME)}.csv'))

        callback_tensorboard = TensorBoard(log_dir=os.path.join(self._log_dir, str(datetime.now())),
                                           histogram_freq=2,
                                           batch_size=32,
                                           write_graph=True,
                                           write_grads=True,
                                           write_images=False)

        # begin training
        t0 = time()
        try:
            self._autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs,
                                  callbacks=[csv_logger, callback_tensorboard],
                                  verbose=False, validation_split=0.1)
        except ValueError:
            self._autoencoder.fit(x, x[-1], batch_size=batch_size, epochs=epochs,
                                  callbacks=[csv_logger, callback_tensorboard],
                                  verbose=False, validation_split=0.1)
        logger.info('Pretraining time: {}'.format(str(time() - t0)))
        self._autoencoder.save(os.path.join(self._save_dir, 'pretrain_cae_model.h5'))

        logger.info('Pretrained weights are saved to {}'.format(os.path.join(self._save_dir,
                                                                             'pretrain_cae_model.h5')))
        self._pretrained = True

    def load_weights(self, weights_path):
        self._model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self._encoder.predict(x)

    def fit(self, x, y=None, batch_size=256, cae_weights=None):
        t0 = time()
        if not self._pretrained and cae_weights is None:
            self.pretrain(x, batch_size, epochs=self._pretrain_epochs)
            self._pretrained = True
        elif cae_weights is not None:
            self._autoencoder.load_weights(cae_weights)
            logger.info('cae_weights is loaded successfully.')

        logger.info('Initializing cluster centers.')
        t1 = time()
        init_variables = self.initialize(x)
        logger.info('Cluster centers initialized: {}'.format(time() - t1))

        import os
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        STRFTIME = "%Y-%m-%d_%H:%M"
        self._logfile = open(os.path.join(self._save_dir, f'train_log_{datetime.now().strftime(STRFTIME)}.csv'), 'w')
        logger.info('Training model.')
        t2 = time()
        self.train_model(x, y, batch_size, init_variables)
        logger.info('Done. {}'.format(time() - t2))

        self._logfile.close()
        logger.info('Saving model to: {}'.format(os.path.join(self._save_dir, 'dcec_model_final.h5')))
        self._model.save_weights(os.path.join(self._save_dir, 'dcec_model_final.h5'))
        t3 = time()
        logger.info('Pretrain time: {}'.format(t1 - t0))
        logger.info('Clustering time: {}'.format(t3 - t1))
        logger.info('Total time: {}'.format(t3 - t0))

    def evaluate(self, y_pred, y, ite, loss):
        acc = np.round(_acc(y, y_pred), 5)
        nmi = np.round(_nmi(y, y_pred), 5)
        ari = np.round(_ari(y, y_pred), 5)
        loss = np.round(loss, 5)
        # logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
        # self._logwriter.writerow(logdict)
        logger.info('Iter: {} Acc: {} nmi: {} ari: {} loss: {}'.format(ite, acc, nmi, ari, loss))

    def stopping_criterion(self, y, y_last):
        delta_label = np.sum(y != y_last).astype(np.float32) / y.shape[0]
        logger.info('delta_label: {}'.format(delta_label))

        if delta_label < self._tol:
            logger.info('delta_label {} < {}'.format(delta_label, self._tol))
            logger.info('Reached tolerance threshold. Stopping training.')
            return True

        return False

    @staticmethod
    def _slice_lists(lsts, index, batch_size, end=False):
        if end:
            return [l[index * batch_size::] for l in lsts]
        else:
            return [l[index * batch_size:(index + 1) * batch_size] for l in lsts]

    @staticmethod
    def _standartize_data(data):
        if isinstance(data, list):
            return data
        else:
            return [data]

    def train_model(self, x, y, batch_size, variables):
        x = DeepClusteringBase._standartize_data(x)
        n_samples = x[0].shape[0]

        logger.info('Update interval {}'.format(self._update_interval))
        save_interval = n_samples / batch_size * 5
        logger.info('Save interval {}'.format(save_interval))

        loss = [0, 0, 0]
        index = 0
        y_pred_last = None
        for ite in range(self._maxiter):

            if ite % self._update_interval == 0:
                logger.info(f'Training model. Iteration #{ite}.')

                self._y_pred, variables = self.update_variables(x, variables)

                if ite > 0:
                    self._logfile.write(str(loss))
                    logger.info('Loss: {}'.format(str(loss)))

                if y is not None:
                    self.evaluate(self._y_pred, y, ite, loss)

                if ite > 0 and self.stopping_criterion(self._y_pred, y_pred_last):
                    break

                y_pred_last = np.copy(self._y_pred)

            if (index + 1) * batch_size > n_samples:
                loss = self._model.train_on_batch(x=DeepClusteringBase._slice_lists(x, index, batch_size, end=True),
                                                  y=DeepClusteringBase._slice_lists(self.signal_example(x, variables),
                                                                                    index,
                                                                                    batch_size,
                                                                                    end=True))
                index = 0
            else:
                loss = self._model.train_on_batch(x=DeepClusteringBase._slice_lists(x, index, batch_size),
                                                  y=DeepClusteringBase._slice_lists(self.signal_example(x, variables),
                                                                                    index, batch_size))
                index += 1

            if ite % save_interval == 0:
                logger.info(
                    'saving model to: {}'.format(os.path.join(self._save_dir, 'dcec_model_{}.h5'.format(str(ite)))))
                self._model.save_weights(os.path.join(self._save_dir, 'dcec_model_{}.h5'.format(str(ite))))

            ite += 1

    def summary(self):
        return self._model.summary()

    def create_model(self):
        pass

    def predict(self, x):
        pass

    def initialize(self, x):
        pass

    def update_variables(self, x, variables):
        pass

    def signal_example(self, x, variables):
        pass


class ClusteringLayer(Layer):
    """
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        # print(type(input_dim), input_dim)
        self.clusters = self.add_weight(shape=(self.n_clusters, int(input_dim)),
                                        initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IDEC(DeepClusteringBase):
    def __init__(self,
                 input_shape,
                 autoencoder_ctor,
                 n_clusters,
                 pretrain_epochs,
                 log_dir,
                 tol=0.05,
                 alpha=1.0,
                 maxiter=1e4,
                 update_interval=140,
                 *args,
                 **kwargs):

        super().__init__(input_shape,
                         autoencoder_ctor,
                         n_clusters,
                         pretrain_epochs,
                         log_dir,
                         tol,
                         *args,
                         **kwargs)

        logger.info(f'Initialized tolerance = {self._tol}.')
        self._update_interval = update_interval
        self._alpha = alpha
        self._maxiter = maxiter

    def create_model(self):
        clustering_layer = ClusteringLayer(self._n_clusters, name='clustering')(self._hidden_layer())
        if type(self._autoencoder.output) is list:
            return Model(inputs=self._autoencoder.input,
                         outputs=[clustering_layer] + self._autoencoder.output)
        else:
            return Model(inputs=self._autoencoder.input,
                         outputs=[clustering_layer,
                                  self._autoencoder.output])

    def score(self, x):
        return self._model.predict(x, verbose=0)[0]  # q

    def predict(self, x):
        return self.score(x).argmax(axis=1)  # q.argmax

    def score_examples(self, x):
        # the clustering layer weights might be considered as the cluster centroids/means/whatever was used for initialization
        cluster_centers_ = self._model.get_layer(name='clustering').get_weights()[0]

        def closeness(x, clusters):
            result = [np.sqrt(sum([(x[i][j] - cluster_centers_[clusters[i]][j]) ** 2
                                   for j in range(len(x[i]))])) for i in range(len(x))]
            return 1. - result / np.max(result)

        x_emb = self.score(x)
        clusters = x_emb.argmax(axis=1)
        return closeness(x_emb, clusters)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, gamma=0.1, loss=['mse'], *args, **kwargs):
        #optimizer = Adam(lr=0.01)
        optimizer='adadelta'
        self._model.compile(loss=['kld'] + loss * (len(self._model.outputs) - 1),  # capture multioutput models
                            loss_weights=[gamma] + [1. for _ in range(len(self._model.outputs)-1)],
                            optimizer=optimizer)

    def initialize(self, x):
        kmeans = KMeans(n_clusters=self._n_clusters, n_init=20, n_jobs=MAX_JOBS)
        self._y_pred = kmeans.fit_predict(self._encoder.predict(x))
        self._model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        return None

    def update_variables(self, x, variables):
        q = self._model.predict(x, verbose=0)[0]
        p = self.target_distribution(q)
        return q.argmax(1), p

    def signal_example(self, x, p):
        # x - list of tensors
        return [p] + x


class IDEC_GMM(IDEC):
    def initialize(self, x):
        gmm = GaussianMixture(n_components=self._n_clusters, n_init=5)
        self._y_pred = gmm.fit_predict(self._encoder.predict(x))
        self._model.get_layer(name='clustering').set_weights([gmm.means_])
        return None

    def score_examples(self, x):
        cluster_centers_ = self._model.get_layer(name='clustering').get_weights()[0]

        def closeness(x, clusters):
            result = [np.sqrt(sum([(x[i][j] - cluster_centers_[clusters[i]][j]) ** 2
                                   for j in range(len(x[i]))])) for i in range(len(x))]
            return result / np.max(result)

        x_emb = self.score(x)
        clusters = x_emb.argmax(axis=1)
        return closeness(x_emb, clusters)


class DAEC(DeepClusteringBase):
    def __init__(self,
                 input_shape,
                 autoencoder_ctor,
                 n_clusters,
                 pretrain_epochs,
                 log_dir,
                 save_dir,
                 tol=1e-3,
                 max_epochs=20,
                 *args,
                 **kwargs):

        super().__init__(input_shape=input_shape, autoencoder_ctor=autoencoder_ctor,
                         n_clusters=n_clusters, pretrain_epochs=pretrain_epochs, log_dir=log_dir, save_dir=save_dir,
                         tol=tol, max_epochs=max_epochs,
                         *args, **kwargs)

    def create_model(self):
        if type(self._autoencoder.output) is list:
            return Model(inputs=self._autoencoder.input,
                         outputs=[self._hidden_layer()] +
                                 self._autoencoder.output)
        else:
            return Model(inputs=self._autoencoder.input,
                         outputs=[self._hidden_layer(),
                                  self._autoencoder.output])

    def predict(self, x):
        return self._kmeans.predict(self._encoder.predict(x, verbose=0))

    def score_examples(self, x):
        def closeness(x, clusters):
            result = [np.sqrt(sum([(x[i][j] - self._kmeans.cluster_centers_[clusters[i]][j]) ** 2
                                   for j in range(len(x[i]))])) for i in range(len(x))]
            return 1. - result / np.max(result)

        x_emb = self._encoder.predict(x, verbose=0)
        clusters = self._kmeans.predict(x_emb)
        return closeness(x_emb, clusters)

    def compile(self, loss_weights=[1., 1.], optimizer='adam', loss=['mse'], *args, **kwargs):
        self._model.compile(loss=['mse'] + loss * (len(self._model.outputs) - 1),  # for multiinput models
                            loss_weights=[loss_weights for _ in range(len(self._model.outputs))],
                            optimizer=optimizer)

    def initialize(self, x):
        pass

    def train_model(self, x, y, batch_size, variables):
        for ite in range(int(self._max_epochs)):
            logger.info('Training k-means...')
            self._kmeans = KMeans(n_clusters=self._n_clusters, n_init=20, n_jobs=MAX_JOBS)
            self._y_pred = self._kmeans.fit_predict(self._encoder.predict(x))
            centroids = self._kmeans.cluster_centers_
            assigned_centroids = np.zeros((x[0].shape[0], centroids.shape[1]))
            for i in range(x[0].shape[0]):
                assigned_centroids[i, :] = centroids[self._y_pred[i]]
            logger.info('Done.')

            if y and ite:
                self.evaluate(self._y_pred, y, ite, (0, 0, 0))

            if ite > 0 and self.stopping_criterion(self._y_pred, self._y_pred_last):
                break

            logger.info(f'Training model. Iteration #{ite}.')
            train_history = self._model.fit(x, [assigned_centroids.tolist(), x[0], x[1], x[2],  # x[3]
                                                ], batch_size=256, verbose=0)

            self._y_pred_last = np.copy(self._y_pred)


class DC_Kmeans(DeepClusteringBase):
    def __init__(self,
                 input_shape,
                 autoencoder_ctor,
                 n_clusters=10,
                 pretrain_epochs=180,
                 *args,
                 **kwargs):
        super().__init__(input_shape, autoencoder_ctor,
                         n_clusters, pretrain_epochs,
                         *args, **kwargs)

        self._lmd = 0.1
        self._ro = 1.

    def create_model(self):

        if type(self._autoencoder.output) is list:
            return Model(inputs=self._autoencoder.input,
                         outputs=[self._hidden_layer()] + self._autoencoder.output)
        else:
            return Model(inputs=self._autoencoder.input,
                         outputs=[self._hidden_layer(),
                                  self._autoencoder.output])

    def get_scores(self):

        def closeness():
            result = [np.sqrt(sum([(self._y[i][j] - self.assigned_centroids[i][j]) ** 2
                                   for j in range(len(self._y[i]))])) for i in range(len(self._y))]
            return 1. - result / np.max(result)

        return closeness()

    def compile(self, loss_weights=[1., 1.], optimizer='adam', loss=['mse'], *args, **kwargs):
        self._model.compile(loss=['mse'] + loss * (len(self._model.outputs) - 1),  # capture multioutput models
                            loss_weights=[loss_weights for _ in range(len(self._model.outputs))],
                            optimizer=optimizer)

    def initialize(self, x):
        self._kmeans = KMeans(n_clusters=self._n_clusters, n_init=20, n_jobs=MAX_JOBS)
        self.f = self._encoder.predict(x)
        self.y_pred = self._kmeans.fit_predict(self.f)
        self.centroids = self._kmeans.cluster_centers_
        self.assigned_centroids = np.zeros((x[0].shape[0], self.centroids.shape[1]))
        for i in range(x[0].shape[0]):
            self.assigned_centroids[i, :] = self.centroids[self.y_pred[i]]
        logger.info('Done.')

        self._y = np.zeros(self.f.shape)
        self.u = np.zeros(self.f.shape)

    def train_model(self, x, y, batch_size, _):
        save_interval = int(x[0].shape[0] / batch_size * 5)
        logger.info('Save interval {}'.format(save_interval))

        for ite in range(int(self._max_epochs)):
            y_pred_last = np.copy(self.y_pred)

            for i in range(x[0].shape[0]):
                self._y[i, :] = (self._lmd * self.assigned_centroids[i] + self._ro * (self.f[i] - self.u[i])) / (
                        self._lmd + self._ro)

            self.u = self.u + self._y - self.f

            for i in range(self._n_clusters):
                self.centroids[i, :] = self.assigned_centroids[self.y_pred == i].mean(axis=0)

            for i in range(x[0].shape[0]):
                self.y_pred[i] = np.linalg.norm(np.ones(self.centroids.shape) * self._y[i] - self.centroids,
                                                axis=1).argmin()

            for i in range(x[0].shape[0]):
                self.assigned_centroids[i, :] = self.centroids[self.y_pred[i]]

            # evaluate the clustering performance
            if y and ite:
                self.evaluate(self.y_pred, y, ite, (0, 0, 0))

            if ite and self.stopping_criterion(self.y_pred, y_pred_last):
                break

            logger.info(f'Training model. Iteration #{ite}.')
            train_history = self._model.fit(x, [self.assigned_centroids.tolist(), x[0], x[1], x[2]], batch_size=128,
                                            verbose=0)
            self.f = self._encoder.predict(x)

            # save intermediate model
            if ite % save_interval == 0:
                # save model checkpoints
                print('saving model to:', self._save_dir + '/model_' + str(ite) + '.h5')
                self._model.save_weights(self._save_dir + '/model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        self._logfile.close()
        print('saving model to:', self._save_dir + '/dcec_model_final.h5')
        self._model.save_weights(self._save_dir + '/dcec_model_final.h5')
