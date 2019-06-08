#!/bin/bash

print_help() {
    echo "Usage : "
    echo "  --hsize=<hsize>        hidden size"
    echo "  --epochs=<epochs>      Epochs to train"
    echo ""
    exit 1
} # end of print_help

unknown_option() {
    echo "Unrecognized option : $1"
    print_help;
    exit 1
} # end of unknow option

#########################################
#
# Default parameters
#
########################################

hsize=16
epochs=200

#########################################
#
# Parse command line arguments
#
########################################
while [ $# -gt 0 ]
do case $1 in
    --hsize=*)          hsize=${1##--hsize=} ;;
    --epochs=*)         epochs=${1##--epochs=} ;;
    *) unknown_option $1 ;;
esac
shift
done

echo "Real datasets"
echo "Time to train 200 epochs with hidden size ${hsize}:"

echo "Cora:"
time_elapsed=`python3 gcn.py --dataset cora        \
                             --n-hidden ${hsize}   \
                             --n-epochs ${epochs}  \
                             --gpu 0               \
                             2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
echo "CiteSeer:"
time_elapsed=`python3 gcn.py --dataset citeseer    \
                             --n-hidden ${hsize}   \
                             --n-epochs ${epochs}  \
                             --gpu 0               \
                             2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
echo "PubMed:"
time_elapsed=`python3 gcn.py --dataset pubmed      \
                             --n-hidden ${hsize}   \
                             --n-epochs ${epochs}  \
                             --gpu 0               \
                             2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
echo

echo "Erdos-Reyi Synthetic graph"
###############################################################################
# Adjust the following three parameters to see how well DGL perform on
# different workload
nodes=32000
p=0.0008
hsize=16
echo "Number of nodes: ${nodes}"
echo "Density: ${p}"
echo "Hidden size: ${hsize}"
time_elapsed=`python3 gcn.py --dataset syn         \
                             --n-hidden ${hsize}   \
                             --n-epochs ${epochs}  \
                             --syn-gnp-n ${nodes}  \
                             --syn-gnp-p ${p}      \
                             --gpu 0               \
                             2>/dev/null | tail -n 1`
echo "${time_elapsed} seconds"
