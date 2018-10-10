import numpy as np
from functools import reduce
from keras.models import Sequential
from keras.layers import Input, Bidirectional, SimpleRNN, GRU, LSTM, Dense, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.generic_utils import to_list
#from keras.engine.topology import Input
#from keras.engine.training import Model
from keras.models import Model
from keras import backend as K
import tensorflow as tf

class BasicModel(object): 
    def build(self, input_shape, bidirectional=True, vat=True):
        
        input_layer = Input(self.input_shape)
        output_layer = self.core_data_flow(input_layer, bidirectional)
        if vat:
            self.model = VATModel(input_layer, output_layer).setup_vat_loss()
        else:
            self.model = Model(input_layer, output_layer)
        return self
    def core_data_flow(self, input_layer, bidirectional):
        x = Dense(160)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if bidirectional:
            x = Bidirectional(SimpleRNN(256))(x)
        else:
            x = SimpleRNN(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return x
    
    def train(self, X_train, X_test, y_train, y_test, batch_size=128, epochs=100, early_stop=False):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), 
                           metrics=['accuracy'])
        np.random.seed(1337)  # for reproducibility
        if early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
        else:
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test)) 
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

class _RNN(BasicModel):
    def __init__(self, bidirectional=True, input_shape=(None,None)):
        self.input_shape = input_shape
        self.model = Sequential()
        self.model.add(Dense(160))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        if bidirectional:
            self.model.add(Bidirectional(SimpleRNN(256)))
        else:
            self.model.add(SimpleRNN(256))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(2))
        self.model.add(Activation('sigmoid'))
    def train(self, X_train, X_test, y_train, y_test, batch_size=128, epochs=100, vat=True,
              early_stop=False):
        orig_loss_func = loss_func = 'categorical_crossentropy'
        if vat:
            loss_func = self.loss_with_vat_loss(loss_func)
            self.model.compile(loss=loss_func, 
                               optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), 
                               metrics=['accuracy'])
            # train, test ともに y, X を横に連結したデータを作ります
            yX_train = np.concatenate([y_train.reshape(1,y_train.shape[0]).T, X_train.reshape(X_train.shape[0], -1)], axis=1)
            yX_test = np.concatenate([y_test.reshape(1,y_test.shape[0]).T, X_test.reshape(X_test.shape[0], -1)], axis=1)
            if early_stop:
                early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
        
                # 普通に学習
                self.model.fit(X_train, yX_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, validation_data=(X_test, yX_test), callbacks=[early_stopping])
            else :
                self.model.fit(X_train, yX_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, validation_data=(X_test, yX_test))
            # 変則的なLossFunctionを与えたので、普通のLossFunctionに変更して再びcompileしないと、evaluate()が失敗します
            self.model.compile(loss=orig_loss_func, optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), metrics=['accuracy'])
        else:
            self.model.compile(loss=loss_func, optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), metrics=['accuracy'])
            if early_stop:
                early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
            
                self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
            else:
                self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, validation_data=(X_test, y_test))
    def loss_with_vat_loss(self, original_loss_func, eps=1, xi=10, ip=1):
        def with_vat_loss(yX_train, y_pred):
            nb_output_classes = y_pred.shape[1]
            y_true = yX_train[:, :nb_output_classes]

            # VAT
            #print (type(yX_train[:, nb_output_classes:]))
            X_train = tf.reshape(yX_train[:, nb_output_classes:], (-1, ) + self.input_shape)
            #X_train = yX_train[:, nb_output_classes:].reshape((-1, ) + self.input_shape)
            d = K.random_normal(X_train.shape)

            for _ in range(ip):
                y = self.core_layers(X_train + self.normalize_vector(d) * xi)
                kld = K.sum(self.kld(y_pred, y))
                d = K.stop_gradient(K.gradients(kld, [d])[0])  # stop_gradient is important!!

            y_perturbation = self.core_layers(X_train + self.normalize_vector(d)*eps)
            kld = self.kld(y_pred, y_perturbation)
            return original_loss_func(y_pred, y_true) + kld
        return with_vat_loss

    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, dim=-1)
        return x / (K.sqrt(z) + K.epsilon())

    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)

                 
class _GRU(BasicModel):
    def __init__(self, bidirectional=True, dropout=True, vat=True):
        self.model = Sequential()
        self.model.add(Dense(160))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())    
        if bidirectional:
            self.model.add(Bidirectional(GRU(256)))
        else:
            self.model.add(GRU(256))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        if vat:
            self.model = VATModel(self.model).setup_vat_loss()
        else:
            self.model = Model(self.model)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), 
                           metrics=['accuracy'])
    def train(self, X_train, X_test, y_train, y_test, batch_size=128, epochs=100,
              early_stop=False):
        if early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(X_test, y_test), callbacks=[early_stopping])
        else:
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                                  validation_data=(X_test, y_test)) 

class _LSTM(BasicModel):
    def __init__(self, bidirectional=True, dropout=True, vat=True):
        self.model = Sequential()
        self.model.add(Dense(160))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        if bidirectional:
            self.model.add(Bidirectional(LSTM(256)))
        else:
            self.model.add(LSTM(256))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        if self.vat:
            self.model = VATModel(self.model).setup_vat_loss()
        else:
            self.model = Model(self.model)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), 
                           metrics=['accuracy'])
    def train(self, X_train, X_test, y_train, y_test, batch_size=128, epochs=100, 
              early_stop=False):
        if early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(X_test, y_test), callbacks=[early_stopping])
        else:
            return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(X_test, y_test)) 
        

