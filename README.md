# Deep Mixture Density Networks for Modeling Market Price Distribution in Online Advertising
This is a repository of experiment code supporting [Deep Mixture Density Networks for Modeling Market Price Distribution in Online Advertising]().

For any problems, please report the issues here.

## Quirk Start

### Requirements

Our experimental platform's key configuration:
* an Intel(R) Xeon(R) E5-2620 v4 CPU processor
* two NVIDIA TITAN Xp GPU processors
* 64 GB memory
* Ubuntu 18.04 operating system
* python 3.7
* cuda 10.2
* tensorflow 2.3
* tensorflow-probability 0.11

Required Python libraries are listed in `./requirements.txt`



### Prepare Dataset
Before run the demo, please first check the [iPinYou dataset](https://contest.ipinyou.com/) and the [criteo dataset](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/).


Then, please create a folder named `datasets`, and unzip the dataset in it.
The file tree looks like this:
```
DMDN
│───README.md
│
└───datasets
│   └───make-ipinyou-data
│       │   1458
│       │   2259
│       │   ...
        make-criteo-data
        |  criteo_attribution_dataset.tsv.gz
...
```

### Run DMDN
Please check the CUDA_VISIBLE_DEVICES variable in `./run_DMDN.sh` before running.

Then run the following code to train and evaluate the DMDN
```bash
bash ./run_DMDN.sh
```
You can find the running logs in this directories `./result/$advertiser/log/run_DMDN`.

Besides, you can find the tensorboard snapshot in `./runs/` in which we put much more useful information for evaluation.