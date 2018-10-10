
# coding: utf-8

# In[1]:


import pandas as pd







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

# Load KDDIdata
#X_kddi, y_kddi = load_kddi_data('./dataset/kddcup99/kddcup.data_10_percent')
X_kddi, y_kddi = load_kddi_data('./dataset/kddcup99/kddcup.data')
#X_kddi


# In[5]:


load_converted_path = './dataset/trafAld.list'
with open(load_converted_path, mode='r') as f:
    s = f.read().replace(' ', ',')
with open(load_converted_path, mode='w') as f:
    f.write(s)
    
X_conv = load_converted_data(load_converted_path)
#X_conv


# In[6]:


import sklearn
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

#pred = KMeans(n_clusters=2, init='random', n_init=10, n_jobs=2).fit_predict(X_kddi)
#acc = accuracy_score(y_kddi, pred)
#print ("Accuracy: {:.3}%".format(acc*100))


# In[7]:


import sklearn
import numpy as np
import pandas as pd
import discriminator
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[8]:


scaler = MinMaxScaler(feature_range=(0, 1))  
#X = scaler.fit_transform(X_kddi)
#X = pd.DataFrame.from_records(X_kddi)
X = np.array(X_kddi.values)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.array(y_kddi.tolist())
#X_conv_test = scaler.fit_transform(X_conv)
#X_conv_test = pd.DataFrame.from_records(X_conv_test)
X_conv_test = np.array(X_conv.values)
X_conv_test = np.reshape(X_conv_test, (X_conv_test.shape[0], X_conv_test.shape[1], 1))


# In[9]:


import discriminator

from keras.utils import np_utils


# In[10]:


my_tsize = .4 # 40%
my_seed = 123
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_tsize, random_state=my_seed)
SAMPLE_SIZE=0
if SAMPLE_SIZE:
    X_train = X_train[:SAMPLE_SIZE]
    y_train = y_train[:SAMPLE_SIZE]
    X_test = X_test[:SAMPLE_SIZE]
    y_test = y_test[:SAMPLE_SIZE]

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


# In[11]:


rnn_type = ['RNN', 'GRU', 'LSTM']
resoult = []
epochs = 100
#clf = discriminator.BasicModel().build(input_shape=(5,5), bidirectional=False, vat=True)
clf = discriminator._RNN(bidirectional=True, input_shape=X_train.shape)
resoult.append(clf.train(X_train, X_test, y_train, y_test, batch_size=128, epochs=epochs, vat=False, early_stop=False))
#score = clf.evaluate(X_test, y_test)#, verbose=0)
score = clf.model.evaluate(X_test, y_test, verbose=0)
print("finish: use_dropout, use_vat: score=%s, accuracy=%s" % (score[0], score[1]))


#clf = discriminator._GRU(bidirectional=True)
#resoult.append(clf.train(X_train, X_test, y_train, y_test, epochs=epochs))
#clf = discriminator._LSTM(bidirectional=True)
#resoult.append(clf.train(X_train, X_test, y_train, y_test, epochs=epochs))


# In[12]:


sco = clf.model.predict(X_conv_test)
print (sco)


# In[ ]:


#x = range(epochs)
#for i in range(len(resoult)):
#    plt.plot(x, resoult[i].history['acc'], label=opt[i])
#plt.title("binary train accuracy")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
#for i in range(len(resoult)):
#    plt.plot(x, resoult[i].history['val_acc'], label=opt[i])
#plt.title("binary test accuracy")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
#for i in range(len(resoult)):
#    plt.plot(x, resoult[i].history['loss'], label=opt[i])
#plt.title("binary train loss")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.ylim([0,1])
#plt.show()
#for i in range(len(resoult)):
#    plt.plot(x, resoult[i].history['val_loss'], label=opt[i])
#plt.title("binary test loss")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.ylim([0,1])
#plt.show()

