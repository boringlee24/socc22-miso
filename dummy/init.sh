#!/bin/bash

python dummy_sender.py --cmd "hostname c0534"
python dummy_sender.py --cmd "config gpu 0 partition 6"
python dummy_sender.py --cmd "logdir static"
