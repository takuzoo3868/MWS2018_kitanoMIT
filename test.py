
# coding: utf-8

# In[1]:


import sklearn, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.backend import tensorflow_backend
import discriminator

### if use gpu, please comment out 
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=config)
#tensorflow_backend.set_session(session)


# In[2]:


def load_kddi_data(data_path):
    col = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
           "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
           "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    dataset = pd.read_csv(data_path, names=col)
    
    # 正常なデータを0, 悪意のあるデータを1とする
    labels = dataset["label"]
    labels = labels.replace({"^.*normal.*":0,"^(?!normal).*$":1}, regex=True)
    # 文字列とラベルを取り除く
    drop_columns = ["protocol_type", "service", "flag", "label"]
    return dataset.drop(drop_columns, axis=1), labels


# In[3]:


def load_converted_data(data_path):
    col = ["num_conn", "startTimet", "orig_pt", "resp_pt", "orig_ht", "resp_ht",
           "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
           "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
           "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    dataset = pd.read_csv(data_path, names=col)

    # 余分な列と文字列を取り除く
    drop_columns = ["num_conn", "startTimet", "orig_pt", "resp_pt", "orig_ht", "resp_ht",
                    "protocol_type", "service", "flag"]
    return dataset.drop(drop_columns, axis=1)


# In[4]:


# Load KDDI Data
X_kddi, y_kddi = load_kddi_data('./kddcup99/kddcup.data_10_percent')
#X_kddi, y_kddi = load_kddi_data('./kddcup99/kddcup.data')


# In[5]:


# Load Converted BOS Data
load_converted_path = './BOS_2014/c11/trafAld.list'
with open(load_converted_path, mode='r') as f:
    s = f.read().replace(' ', ',')
with open(load_converted_path, mode='w') as f:
    f.write(s)
X_conv = load_converted_data(load_converted_path)


# In[6]:


scaler = MinMaxScaler(feature_range=(0, 1))  
#X = scaler.fit_transform(X_kddi)
X = np.array(X_kddi.values)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.array(y_kddi.tolist())
#X_conv_test = scaler.fit_transform(X_conv)
X_conv_test = np.array(X_conv.values)
X_conv_test = np.reshape(X_conv_test, (X_conv_test.shape[0], X_conv_test.shape[1], 1))


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

SAMPLE_SIZE = 0
if SAMPLE_SIZE:
    X_train = X_train[:SAMPLE_SIZE]
    y_train = y_train[:SAMPLE_SIZE]
    X_test = X_test[:SAMPLE_SIZE]
    y_test = y_test[:SAMPLE_SIZE]

#y_train = np_utils.to_categorical(y_train, 2)
#y_test = np_utils.to_categorical(y_test, 2)


# In[ ]:


rnn_type = ['RNN', 'GRU', 'LSTM']
training_resoult = []
predict_resoult = []
batch_size = 128
epochs = 100
base = discriminator.BasicModel()

for i in rnn_type:
    clf = base.build(input_shape=(X_train.shape[1], 1), rnn_type=i, bidirectional=True, vat=True)
    training_resoult.append(clf.train(X_train, X_test, y_train, y_test, batch_size=batch_size,
                                      epochs=epochs, early_stop=False))
    predict_resoult.append(np.round(clf.model.predict(X_conv_test)))

#score = clf.evaluate(X_test, y_test)#, verbose=0)
#score = clf.model.evaluate(X_test, y_test, verbose=0)
#print("finish: use_dropout, use_vat: score=%s, accuracy=%s" % (score[0], score[1]))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = range(epochs)
for i in range(len(training_resoult)):
    plt.plot(x, training_resoult[i].history['acc'], label=rnn_type[i])
plt.title("binary train accuracy")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
for i in range(len(training_resoult)):
    plt.plot(x, training_resoult[i].history['val_acc'], label=rnn_type[i])
plt.title("binary test accuracy")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
for i in range(len(training_resoult)):
    plt.plot(x, training_resoult[i].history['loss'], label=rnn_type[i])
plt.title("binary train loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim([0,1])
plt.show()
for i in range(len(training_resoult)):
    plt.plot(x, training_resoult[i].history['val_loss'], label=rnn_type[i])
plt.title("binary test loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim([0,1])
plt.show()


# In[ ]:


predict_resoult[0].hist(alpha=.5,label=0,bins=40)
predict_resoult[0].hist(alpha=.5,label=1,bins=40)
plt.legend()

