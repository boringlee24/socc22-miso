#!/bin/bash
sudo nvidia-smi -i $1 -pm ENABLED &&

sudo nvidia-smi -i $1 -mig 1

#sudo nvidia-smi mig -i $1 -lgip &&
#
#sudo nvidia-smi mig -i $1 -cgi 19 &&
#
#sudo nvidia-smi mig -i $1 -lcip &&
#
#sudo nvidia-smi mig -i $1 -gi 9 -cci

#sudo nvidia-smi -i 0 -pm ENABLED &&
#
#sudo nvidia-smi -i 0 -mig 1 &&
#
#sudo nvidia-smi mig -i 0 -lgip &&
#
#sudo nvidia-smi mig -i 0 -cgi 19 &&
#
#sudo nvidia-smi mig -i 0 -lcip &&
#
#sudo nvidia-smi mig -i 0 -gi 11 -cci

