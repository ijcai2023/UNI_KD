# Distilling Universal and Joint Knowledge for Cross-Domain Model Compression on Time Series Data

**UNI_KD** is a PyTorch implementation for Distilling Universal and Joint Knowledge for Cross-Domain Model Compression on Time Series Data. 

# Disclaimer: This work is for research only and not for commercial use!

## Requirmenets:
- Python3
- Pytorch==1.7
- Numpy==1.20.1
- scikit-learn==0.24.1
- Pandas==1.2.4
- skorch==0.10.0 (For DEV risk calculations)
- openpyxl==3.0.7 (for classification reports)
- Wandb=0.12.7 (for sweeps)

## Datasets

### Available Datasets
We used four public datasets in this study. We also provide the **preprocessed** versions as follows:

- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [FD](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download)
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)

Please download these datasets and put them in the respective folder in "data"


## Unsupervised Domain Adaptation Algorithms
### Existing Benchmark Algorithms
- [MMDA](https://arxiv.org/abs/1901.00282)
- [DANN](https://arxiv.org/abs/1505.07818)
- [CDAN](https://arxiv.org/abs/1705.10667)
- [DIRT-T](https://arxiv.org/abs/1802.08735)
- [HoMM](https://arxiv.org/pdf/1912.11976.pdf)
- [DDC](https://arxiv.org/abs/1412.3474)
- [CoDATS](https://arxiv.org/pdf/2005.10996.pdf)
- [JKU](https://arxiv.org/pdf/2005.07839.pdf)
- [AAD](https://arxiv.org/pdf/2010.11478.pdf)
- [MobileDA](https://ieeexplore.ieee.org/abstract/document/9016215/)

## Runing Benchmark Algorithms 

### SOTA UDA methods (MMDA,DDC,et al.)

To train a model with UDA methods:

```
python uda_benchmark.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method DANN \
                --dataset HAR \
                --backbone CNN \
                --num_runs 3 \
```

### UDA+KD methods (JKU, MobileDA, AAD)

To train a model with UDA methods:

```
python jku_mobileda_aad.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method JointUKD \
                --dataset HAR \
                --backbone CNN \
                --num_runs 3 \
```

## Runing Proposed UDA_KD Algorithm

### Teacher training
Our approach requires a pre-trained teacher. We utilize DANN method to train a teacher and store them in 'experiments_logs/HAR/Teacher_CNN'.
For different dataset, please save the teachers into respectively dataset folder. Note that for teacher, we set 'feature_dim = 64' in 'configs/data_model_configs.py' 
and for the student we set 'feature_dim = 16'.

## Student training
To train a student with our proposed approach, run:

```
python proposed_uda_kd.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method UDA_KD \
                --dataset HAR \
                --backbone CNN \
                --num_runs 3 \
```

## Claims
Part of benchmark methods code are from [AdaTime](https://github.com/emadeldeen24/AdaTime)
