# MWS2018_kitanoMIT
MWScup2018 事前課題  

データセット等で収集されているpcapファイルをkddcup99形式へ変換し，

- VATアルゴリズムを用いたクラスタリング
- ipアドレス情報を基にした世界地図へのマッピング

等，pcapを用いたデータ解析を補助するツールです．

## requirement
pyenv + virtualenv

```
$ pyenv virtualenv 3.7.0 mws2018
$ git clone <this repo url> <install path>
$ cd <install path>
$ pyenv local mws2018
$ pip install -r requirements.txt
$ sh setup.sh
```
注：pcap to kddcup99にbroを使用します。Anacondaがインストールされていると、broのインストールに失敗する可能性があります。

## usage 
基本的な利用方法は以下の通りです．

pcapファイルからkddcup99形式への変換 (最終的に"trafAld.list"というファイルができます)
```bash
$ bro -r <pcap file path> tcpdump2gureKDDCup99/darpa2gurekddcup.bro > conn.list
$ sort -n conn.list > conn_sort.list
$ tcpdump2gureKDDCup99/trafAld.out conn_sort.list
```

全結合ネットワークによる学習、および予測
```bash
$ python attack_classification_and_mapping.py --mode train
$ python attack_classification_and_mapping.py --mode predict --file_path trafAld.list
```

`--mode predictを実行するとマッピングデータが"./trafAld-map.html"にエクスポートされます．`

|     option    |             description              |
|:-------------:|:--------------------------------------:|
| --multi_class | 良性悪性クラスから多クラスへ拡張します |
| --use_gpu     | 解析にGPUを用いる場合に使用します      |
