#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sklearn, argparse, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.backend import tensorflow_backend
import discriminator, geoip2.database, folium
import warnings
warnings.filterwarnings('ignore')


def load_kddi_data(file_path, multi_class):
    """ Load KDDI Cup 99 Data
    """
    col = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
           "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
           "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    dataset = pd.read_csv(file_path, names=col)
    
    labels = dataset["label"]
    if multi_class:
        # Replace label to Benign: 0, Probe: 1, DoS: 2, U2R: 3, R2L: 4
        # Benign: 通常のコネクション, Probe: 攻撃対象の探索・調査, DoS: DoS攻撃
        # U2R: ローカルマシンからrootへの許可されていないアクセス
        # R2L: リモートマシンからの許可されていないアクセス
        labels = labels.replace({"^.*normal.*":0,"^.*ipsweep.*":1,"^.*nmap.*":1, "^.*portsweep.*":1,
                                 "^.*satan.*":1,"^.*mscan.*":1,"^.*saint.*":1,"^.*back.*":2,"^.*land.*":2,
                                 "^.*neptune.*":2,"^.*pod.*":2,"^.*smurf.*":2,"^.*teardrop.*":2,
                                 "^.*mailbomb.*":2,"^.*apache2.*":2,"^.*processtable.*":2,"^.*udpstorm.*":2,
                                 "^.*buffer_overflow.*":3,"^.*loadmodule.*":3,"^.*perl.*":3,"^.*rootkit.*":3,
                                 "^.*httptunnel.*":3,"^.*xterm.*":3,"^.*ps.*":3,"^.*worm.*":3,
                                 "^.*ftp_write.*":4,"^.*guess_passwd.*":4,"^.*imap.*":4,"^.*multihop.*":4,
                                 "^.*phf.*":4,"^.*spy.*":4,"^.*warezclient.*":4,"^.*warezmaster.*":4,
                                 "^.*snmpgetattack.*":4,"^.*snmpguess.*":4,"^.*xsnoop.*":4,
                                 "^.*named.*":4,"^.*sendmail.*":4,"^.*sqlattack.*":4,"^.*xlock.*":4}, regex=True)
    else:
        # Replace label to Benign: 0, Malicious: 1
        labels = labels.replace({"^.*normal.*":0,"^(?!normal).*$":1}, regex=True)
        
    dataset["protocol_type"] = dataset["protocol_type"].replace({"^.*tcp*":0,"^.*udp*$":1,
                                                                 "^.*icmp*$":2}, regex=True)
    dataset["flag"] = dataset["flag"].replace({"^.*OTH*":0,"^.*SF*$":1,"^.*SH*$":2,"^.*S0*$":3,"^.*REJ*$":4,
                                               "^.*RSTR*$":5,"^.*RSTO*$":6, "^.*S2*$":7,"^.*S1*$":8,
                                               "^.*S3*$":7,"^.*SHR*$":8,"^.*RSTRH*$":9}, regex=True)
    
    # Drop columns 
    drop_columns = ["protocol_type", "service", "label"]
    return dataset.drop(drop_columns, axis=1), labels


def load_converted_data(file_path):
    """ Load converted pcap data
    """
    col = ["num_conn", "startTimet", "orig_pt", "resp_pt", "orig_ht", "resp_ht",
           "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
           "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
           "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    dataset = pd.read_csv(file_path, names=col)
    
    dataset["protocol_type"] = dataset["protocol_type"].replace({"^.*tcp*":0,"^.*udp*$":1,
                                                                 "^.*icmp*$":2}, regex=True)
    dataset["flag"] = dataset["flag"].replace({"^.*OTH*":0,"^.*SF*$":1,"^.*SH*$":2,"^.*S0*$":3,"^.*REJ*$":4,
                                               "^.*RSTR*$":5,"^.*RSTO*$":6, "^.*S2*$":7,"^.*S1*$":8,
                                               "^.*S3*$":7,"^.*SHR*$":8,"^.*RSTRH*$":9}, regex=True)
    
    # orig_ht: 送信元ip, resp_ht: 送信先ip
    orig_ip_list = dataset["orig_ht"]
    resp_ip_list = dataset["resp_ht"]
    drop_columns = ["num_conn", "startTimet", "orig_pt", "resp_pt", "orig_ht", "resp_ht", "protocol_type",
                    "service"] 
    return dataset.drop(drop_columns, axis=1), orig_ip_list, resp_ip_list


def train(multi_class, use_gpu):
    """ Training from KDDI Cup 99 data 
    """
    if use_gpu:
        # Set GPU
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        tensorflow_backend.set_session(session)
        
    # Load KDDI Data
    X_kddi, y_kddi = load_kddi_data(file_path='./kddcup99/kddcup.data_10_percent', multi_class=multi_class)
    
    # Preprocess for data
    split_size = .4   # split 40% of the data for test
    scaler = MinMaxScaler(feature_range=(0, 1))  
    X = scaler.fit_transform(X_kddi)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.array(y_kddi.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=123)

    # If training is slow, please set SAMPLE_SIZE
    SAMPLE_SIZE = 0
    if SAMPLE_SIZE:
        X_train = X_train[:int(SAMPLE_SIZE*(1-split_size))]
        y_train = y_train[:int(SAMPLE_SIZE*(1-split_size))]
        X_test = X_test[:int(SAMPLE_SIZE*split_size)]
        y_test = y_test[:int(SAMPLE_SIZE*split_size)]
    
    if multi_class:
        y_train = np_utils.to_categorical(y_train, 5)
        y_test = np_utils.to_categorical(y_test, 5)
        
    # Train
    batch_size = 128
    epochs = 100
    nn_type = 'Dense'
    os.makedirs('save_data', exist_ok=True)
    if multi_class:
        save_name = 'save_data/'+nn_type+'_weights_multi.h5'
    else:
        save_name = 'save_data/'+nn_type+'_weights.h5'
    base = discriminator.BasicModel(multi_class)
    clf = base.build(input_shape=(39, 1), nn_type=nn_type, vat=True)
    clf.train(X_train, X_test, y_train, y_test, batch_size=batch_size, epochs=epochs, early_stop=True)
    clf.model.save_weights(save_name)
        
        
def predict(multi_class, use_gpu, file_path):
    if use_gpu:
        # Set GPU
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        tensorflow_backend.set_session(session)
        
    # Load converted pcap data
    with open(file_path, mode='r') as f:
        s = f.read().replace(' ', ',')
    with open(file_path, mode='w') as f:
        f.write(s)
    X_converted, orig_ip_list, resp_ip_list = load_converted_data(file_path=file_path)
    
    # Preprocess for data
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    X_test_nolabel = scaler.fit_transform(X_converted)
    X_test_nolabel = np.reshape(X_test_nolabel, (X_test_nolabel.shape[0], X_test_nolabel.shape[1], 1))
    
    # Predict
    nn_type = 'Dense'
    if multi_class:
        save_name = 'save_data/'+nn_type+'_weights_multi.h5'
    else:
        save_name = 'save_data/'+nn_type+'_weights.h5'
    base = discriminator.BasicModel(multi_class)
    clf = base.build(input_shape=(39, 1), nn_type=nn_type, vat=True)
    clf.model.load_weights(save_name)
    predict_resoult = clf.model.predict(X_test_nolabel)
    p_resoult = []
    for i in range(len(predict_resoult)):
        if multi_class:
            p_resoult.append(predict_resoult[i].argmax())  
        else:
            for j in range(len(predict_resoult[i])):                                             
                p_resoult.append(int(np.round(predict_resoult[i][j])))
    
    # Count appered ip address
    connection_list = []
    temp = []
    appered_counta = []
    for i in range(len(orig_ip_list)):
        connection_list.append([orig_ip_list[i], resp_ip_list[i], p_resoult[i]])
    for i in connection_list: 
        if not i in temp:
            temp.append(i)
            appered_counta.append([i, 1])
        else:
            for j in range(len(appered_counta)):
                if appered_counta[j][0] == i:
                    appered_counta[j][1]+=1;
                    break
                    
    # Search ip info
    orig_record = []
    resp_record = []
    for i in range(len(temp)):
        orig_record.append(search_ip_info(ip=appered_counta[i][0][0]))
        resp_record.append(search_ip_info(ip=appered_counta[i][0][1]))
    # Print originator and responder
    attack_type_multi = ['Benign', 'Probe', 'DoS', 'U2R', 'R2L']
    attack_type = ['Benign', 'Malisious']
    for i in range(len(orig_record)):
        try:
            print('orig: '+appered_counta[i][0][0]+'\t'+orig_record[i].city.name, end="\t")
            try:
                print('resp: '+appered_counta[i][0][1]+'\t'+resp_record[i].city.name, end="\t")
            except:
                print('resp: '+appered_counta[i][0][1]+'\t'+'Private IP', end="\t")
            if multi_class:
                print('type: '+attack_type_multi[appered_counta[i][0][2]]+'\t'+str(appered_counta[i][1])+' times')
            else:
                 print('type: '+attack_type[appered_counta[i][0][2]]+'\t'+str(appered_counta[i][1])+' times')
        except:
            print('orig: '+appered_counta[i][0][0]+'\t'+'Private IP', end="\t")
            try:
                print('resp: '+appered_counta[i][0][1]+'\t'+resp_record[i].city.name, end="\t")
            except:
                print('resp: '+appered_counta[i][0][1]+'\t'+'Private IP', end="\t")
            if multi_class:
                print('type: '+attack_type_multi[appered_counta[i][0][2]]+'\t'+str(appered_counta[i][1])+' times')
            else:
                 print('type: '+attack_type[appered_counta[i][0][2]]+'\t'+str(appered_counta[i][1])+' times')
            
    # Make ip map
    make_map(multi_class=multi_class, appered_counta=appered_counta,
             orig_record=orig_record, resp_record=resp_record, file_path=file_path)
    

def search_ip_info(ip):
    # Load geoip database
    reader = geoip2.database.Reader('./Geoip/GeoLite2-City.mmdb')    
    try:
        return reader.city(ip)
    except:
        return 'Private IP'


def make_map(multi_class, appered_counta, orig_record, resp_record, file_path):
    #  Benign: '#0000ff', Probe: '#ff4500', DoS: '#008000', U2R: '#ffa500', R2L: '#ee82ee', Malicious: '#dc143c'
    color_list_multi = ['#0000ff', '#ff4500', '#008000', '#ffa500', '#ee82ee']
    color_list = ['#0000ff', '#dc143c']
    RADIUS_WEIGHT = 7
    
    ip_map = folium.Map(location=[30, 0], zoom_start=3)
    for i in range(len(appered_counta)):
        if multi_class:
            try:
                folium.vector_layers.CircleMarker(
                    location=[resp_record[i].location.latitude+0.001*appered_counta[i][0][2],
                        resp_record[i].location.longitude+0.001*appered_counta[i][0][2]],
                    popup=appered_counta[i][0][1],
                    radius=appered_counta[i][1]*RADIUS_WEIGHT,
                    color=color_list_multi[appered_counta[i][0][2]],
                    fill_color=color_list_multi[appered_counta[i][0][2]]
                ).add_to(ip_map)
            except:
                pass
        else:
            try:
                folium.vector_layers.CircleMarker(
                    location=[resp_record[i].location.latitude+0.001*appered_counta[i][0][2],
                        resp_record[i].location.longitude+0.001*appered_counta[i][0][2]],
                    popup=appered_counta[i][0][1],
                    radius=appered_counta[i][1]*RADIUS_WEIGHT,
                    color=color_list[appered_counta[i][0][2]], fill_color=color_list[appered_counta[i][0][2]]
                ).add_to(ip_map)
            except:
                pass
    
    if multi_class:
        legend_html =   '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 130px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        ">&nbsp; Connection Type <br>
                          &nbsp; Benign &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #0000ff"></i><br>
                          &nbsp; Probe &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #ff4500"></i><br>
                          &nbsp; DoS &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #008000"></i><br>
                          &nbsp; U2R &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #ffa500"></i><br>
                          &nbsp; R2L &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #ee82ee"></i>
            </div>
            ''' 
        ip_map.get_root().html.add_child(folium.Element(legend_html))
        ip_map.save(os.path.splitext(file_path)[0]+'-multi-map.html')
    else:
        legend_html =   '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 70px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        ">&nbsp; Connection Type <br>
                          &nbsp; Benign &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #0000ff"></i><br>
                          &nbsp; Malicious &nbsp; <i class="fa fa-circle-o fa-lg" style="color: #dc143c"></i>
            </div>
            ''' 
        ip_map.get_root().html.add_child(folium.Element(legend_html))
        ip_map.save(os.path.splitext(file_path)[0]+'-map.html')
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="select mode: 'train' or 'predict'")
    parser.add_argument("--multi_class", dest="multi_class", action="store_true",
                        help="use_flag for multi_class on clustering")
    parser.set_defaults(multi_class=False)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", help="use_flag for gpu")
    parser.set_defaults(use_gpu=False)
    parser.add_argument("--file_path", type=str, default='_', help="file_path for predict")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(multi_class=args.multi_class, use_gpu=args.use_gpu)
    elif args.mode == "predict":
        if args.file_path == '_':
            print('Please enter "--file_path FILEPATH"')
        else:
            predict(multi_class=args.multi_class, use_gpu=args.use_gpu, file_path=args.file_path)
