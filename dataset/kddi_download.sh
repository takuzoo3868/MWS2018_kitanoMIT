#!/bin/bash
wget -r -l 1 http://kdd.ics.uci.edu/databases/kddcup99/
gunzip -r kdd.ics.uci.edu/
ln -s kdd.ics.uci.edu/databases/kddcup99 kddcup99
