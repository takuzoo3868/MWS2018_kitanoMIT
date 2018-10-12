import numpy as np, tensorflow as tf
from functools import reduce
from keras.layers import Bidirectional, GRU, LSTM, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.utils.generic_utils import to_list
from keras import backend as K


class BasicModel(object): 
    def __init__(self, multi_class=False):
        self.multi_class = multi_class
        
    def build(self, input_shape, nn_type='Dense', bidirectional=True, vat=True):
        """ build model
        :param input_shape: shape=(number of input rows, 1)
        :param nn_type: select 'Dense' or 'RNN' or 'GRU' or 'LSTM'
        :param bidirectional: use_flag for Bidirectional rnn
        :param vat: use_flag for VAT
        :param multi_class: use_flag for multi_class
        :return: self
        """
        input_layer = Input(input_shape)
        output_layer = self.core_data_flow(input_layer, nn_type, bidirectional)
        if vat:
            self.model = VATModel(input_layer, output_layer).setup_vat_loss()
        else:
            self.model = Model(input_layer, output_layer)
        return self
    
    def core_data_flow(self, input_layer, nn_type, bidirectional):
        """ build nn model
        :param input_layer: required for Model()
        :return: layer
        """
        if nn_type == 'Dense':
            x = Dense(64, activation='relu')(input_layer)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Flatten()(x)
            if self.multi_class:
                x = Dense(5, activation='softmax')(x)
            else:
                x = Dense(1, activation='sigmoid')(x)
        else:
            x = Dense(160)(input_layer)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if nn_type == 'RNN':
                x = Bidirectional(SimpleRNN(256))(x)
            elif nn_type == 'GRU':
                x = Bidirectional(GRU(256))(x)
            elif nn_type == 'LSTM':
                x = Bidirectional(LSTM(256))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.multi_class:
                x = Dense(5, activation='softmax')(x)
            else:
                x = Dense(1, activation='sigmoid')(x)
        return x
    
    def train(self, X_train, X_test, y_train, y_test, batch_size=128, epochs=100, early_stop=True):
        """ train rnn model
        :param X_train, X_test, y_train, y_test: X is feature vectol. y is label
        :param batch_size: onece per training size
        :param epochs: number of iterations
        :param early_stopinput_layer: use_flag for EarlyStopping
        :return: history data
        """
        if self.multi_class:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                               metrics=['accuracy'])
        else:
            self.model.compile(loss='binary_crossentropy',
                               optimizer='RMSprop',
                               metrics=['accuracy'])
        
        np.random.seed(1337)  # for reproducibility
        if early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=5)
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(X_test, y_test), callbacks=[early_stopping])
        else:
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(X_test, y_test)) 
        
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        return self._score(y, self.model.predict_proba(X)[:, 1])
    def _score(self, true_label, predicted_prob):
        """ calculate the performance score for binary calssification
        :param true_label: the ground truth score
        :param predicted_label: the predicted probability
        :return: a dict of scores
        """
        score_dict = dict()
        score_dict['AUC'] = metrics.roc_auc_score(true_label, predicted_prob)
        predicted_label = [0 if prob < 0.5 else 1 for prob in predicted_prob]
        score_dict['Accuracy'] = metrics.accuracy_score(true_label, predicted_label)
        cm = metrics.confusion_matrix(true_label, predicted_label)
        score_dict['Confusion Matrix'] = cm
        score_dict['TPR'] = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
        score_dict['FPR'] = cm[0, 1] / float(cm[0, 0] + cm[0, 1])
        return score_dict
        
class VATModel(Model):
    """ VAT (Virtual Adversarial Training)
        refored https://qiita.com/mokemokechicken/items/b3cb3d65b6876ccf1a5b
    """
    _vat_loss = None

    def setup_vat_loss(self, eps=1, xi=10, ip=1):
        self._vat_loss = self.vat_loss(eps, xi, ip)
        return self
    
    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self._vat_loss is not None:
            losses += [self._vat_loss]
        return losses
    
    def vat_loss(self, eps, xi, ip):
        normal_outputs = [K.stop_gradient(x) for x in to_list(self.outputs)]
        d_list = [K.random_normal(tf.shape(x)) for x in self.inputs]
        for _ in range(ip):
            new_inputs = [x + self.normalize_vector(d)*xi for (x, d) in zip(self.inputs, d_list)]
            new_outputs = to_list(self.call(new_inputs))
            klds = [K.sum(self.kld(normal, new)) for normal, new in zip(normal_outputs, new_outputs)]
            kld = reduce(lambda t, x: t+x, klds, 0)
            d_list = [K.stop_gradient(d) for d in K.gradients(kld, d_list)]
        new_inputs = [x + self.normalize_vector(d) * eps for (x, d) in zip(self.inputs, d_list)]
        y_perturbations = to_list(self.call(new_inputs))
        klds = [K.mean(self.kld(normal, new)) for normal, new in zip(normal_outputs, y_perturbations)]
        kld = reduce(lambda t, x: t + x, klds, 0)
        return kld
    
    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, axis=-1)
        return x / (K.sqrt(z) + K.epsilon())
    
    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)
