# MEnsA: Mix-up Ensemble Average for UnsupervisedMulti Target Domain Adaptation on 3D Point Clouds

[Ashish Sinha](https://sinashish.github.io) | [Jonghyun Choi](https://ppolon.github.io) | [Paper](#) | [Arxiv](#) 

Accepted as poster at [Continual Learning Workshop](https://sites.google.com/view/clvision2023/overview?authuser=0), [CVPR 2023](https://cvpr.thecvf.com/)

## Abstract

Unsupervised domain adaptation (UDA) addresses the problem of distribution shift between the unlabelled target domain and labelled source domain. While the single target domain adaptation (STDA) is well studied in the literature for both 2D and 3D vision tasks, multi-target domain adaptation (MTDA) is barely explored for 3D data despite its wide real-world applications such as autonomous driving systems for various geographical and climatic conditions. We establish an MTDA baseline for 3D point cloud data by proposing to mix the feature representations from all domains together to achieve better domain adaptation performance by an ensemble average, which we call **M**ixup **Ens**emble **A**verage or **MEnsA**. With the mixed representation, we use a domain classifier to improve at distinguishing the feature representations of source domain from those of target domains in a shared latent space. In empirical validations on the challenging PointDA-10 dataset, we showcase a clear benefit of our simple method over previous unsupervised STDA and MTDA methods by large margins (up to 17.10% and 4.76% on averaged over all domain shifts).

## Pipeline Overview
![image](assets/overview.png)

## Proposed Mixup Schema
![mixup](assets/mtda_schema.png)

## 3D Point Cloud MTDA Results 
![results](assets/results.png)

## Ablation wrt $\mathcal{L}$
![ablation](assets/abln.png)

## Repo Structure
```
.
├── assets                  # paper figures
├── data                    # root dir for data
│   └── PointDA10_data
│       ├── ModelNet10
│       ├── ScanNet
│       └── ShapeNet
├── Dockerfile              # dockerfile for building the container
├── main.py                 # training script
├── saved
│   ├── ckpt                # tensorboard logs
│   └── logs                # model checkpoints
├── models                  # the models
│   ├── ...
├── prepare_dataset.sh      # fetch dataset
├── README.md
├── requirements.txt        
└── src                     # trainer and utility functions
    └── ...
```

## Dependecies
- CUDA:10.2
- CUDNN:7.0
- Python3
- Pytorch:1.7.1
  
## Preparing the Dataset

We use the the benchmark point cloud dataset for domain adaptation – `PointDA-10`  for experimentation.
To download the dataset, and prepare the folder structure. Simplya run,

```bash
bash prepare_dataset.sh
```
## Running the code

To run the code, for training with default parameters, simply run

```bash
python3 main.py
```

To train a new model with changed hyperparameters, follow this:

```bash
python3 main.py -s <SOURCE DATASET>\
                -e <EPOCHS>\
                -b <BATCH SIZE>\
                -g <GPU IDS>\
                -lr <LEARNING RATE>\
                -mixup <SWITCH MIXING>\
                -mix_sep <TO USE BASELINE MIXUP: SEP>\
                -mix_type <MIXUP VARIANTS: {
                    -1 : MEnsA,
                    0 : Mixup A,
                    1 : Mixup B,
                    2 : Mixup C
                }>\
                -seed <SEED VALUE>\
                -r <CHECKPOINT PATH FOR RESUME TRAINING>\
                -log_interval <LOGGING INTERVAL>\
                -save_interval <SAVE INTERVAL>\
                -datadir <PATH TO DATA>\
                -lambda_mix <LOSS WEIGHT FOR MIXING>\
                -lambda_adv <LOSS WEIGHT FOR ADVERSERIAL LOSS>\
                -lambda_mmd <LOSS WEIGHT FOR MMD LOSS>\
                -gamma <GAMMA WEIGHT>
```

## Running the code in docker

- Create the docker container
    ```bash
    docker build -f Dockerfile -t mtda:pc .
    ```
- Enter the container and mount the dataset 
    ```bash
    docker run -it --gpus all -v </path/to/dataset/>:/data mtda:pc
    ```
- Run the code inside the container
    ```bash
    CUDA_VISIBLE_DEVICES=<GPU ID> python3 main.py -g <GPU ID> -s <SOURCE DATASET> -mixup 
    ```

## Pretrained Models

The pretrained models for the experiments are available [here](https://drive.google.com/drive/folders/183RIEz7IpesWSk39rCBIbrg4Wn1Yj62L?usp=sharing).

## Acknowledgements

Some of the code is borrowed from [PointDAN](https://github.com/canqin001/PointDAN). 