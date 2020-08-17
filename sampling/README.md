## Distributed training

This is an example of training GraphSage in a distributed fashion. To train GraphSage, it has four steps:

### Step 1: partition the graph.

The example provides a script to partition some builtin graphs such as Reddit and OGB product graph.
If we want to train GraphSage on 4 machines, we need to partition the graph into 4 parts.

In this example, we partition the OGB product graph into 4 parts with Metis. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
# partition graph
python3 partition_graph.py --dataset ogb-product --num_parts 4 --balance_train --balance_edges
```

### Step 2: copy the partitioned data to the cluster

When copying data to the cluster, we recommend users to copy the partitioned data to NFS so that all worker machines
will be able to access the partitioned data.

### Step 3: run servers

To perform actual distributed training (running training jobs in multiple machines), we need to run
a server on each machine. Before running the servers, we need to update `ip_config.txt` with the right IP addresses.

On each of the machines, set the following environment variables.

```bash
export DGL_ROLE=server
export DGL_IP_CONFIG=ip_config.txt
export DGL_CONF_PATH=data/ogb-product.json
export DGL_NUM_CLIENT=2
```

```bash
# run server on machine 0
export DGL_SERVER_ID=0
python3 mp_dataloader.py
```

### Step 4: run trainers
We run a trainer process on each machine. Here we use Pytorch distributed. We need to use pytorch distributed launch to run each trainer process.
Pytorch distributed requires one of the trainer process to be the master. Here we use the first machine to run the master process.

```bash
# run client on machine 0
export OMP_NUM_THREADS=1
python3 mp_dataloader.py --conf_file data/ogb-product.json --ip_config ip_config.txt --batch-size 1000 --num-workers 1
```

## Pytorch DataLoader

```bash
python3 local_dataloader.py --batch-size 1000 --num-workers 1
```
