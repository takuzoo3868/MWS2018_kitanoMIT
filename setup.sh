#!/bin/bash
sudo apt-get install -y wget git cmake make gcc g++ flex bison libpcap-dev libssl-dev python-dev swig zlib1g-dev
git clone --recursive git://git.bro.org/bro
cd bro
./configure
make -j4
sudo make install
echo "export PATH=$PATH:/usr/local/bro/bin" > ~/.profile
source ~/.profile
cd ..
git clone https://github.com/inigoperona/tcpdump2gureKDDCup99.git
gcc tcpdump2gureKDDCup99/trafAld.c -o tcpdump2gureKDDCup99/trafAld.out

cd dataset
bash "kddi_download.sh"
cd ..

cd Geoip
bash "geoiplite_download.sh"
cd ..

