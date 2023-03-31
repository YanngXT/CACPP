# CACPP

## Introduction

This repository contained code for "CACPP: a contrastive learning-based siamese network to identify anti-cancer peptides based on sequence only".

Any question is welcomed to be asked by issue and I will try my best to solve your problems.

## Paper Abstract

The anti-cancer peptide (ACP) recently has been receiving increasing attention in cancer therapy due to its low consumption, few adverse side effects, and easy accessibility. However, it remains a great challenge to identify anti-cancer peptides via experimental approaches, requiring expensive and time-consuming experimental studies. In addition, traditional machine-learning-based methods are proposed for ACP prediction mainly depending on hand-crafted feature engineering, which normally achieves low prediction performance. In this study, we propose CACPP(Contrastive ACP Predictor), a deep learning framework based on the convolutional neural network (CNN) and contrastive learning for accurately predicting anti-cancer peptides. In particular, we introduce the TextCNN model to extract the high-latent features based on the peptide sequences only and exploit the contrastive learning module to learn more distinguishable feature representations to make better predictions. Comparative results on the benchmark datasets indicates that CACPP outperforms all the state-of-the-art methods in the prediction of anti-cancer peptides. Moreover, to intuitively show that our model has good classification ability, we visualized the dimension reduction of the features from our model and explored the relationship between peptide sequences and anti-cancer functions. Furthermore, we also discussed the influence of dataset construction on model prediction and explored our model performance on the datasets with verified negative samples.

# How to use it

In banch [master](https://github.com/yanngfengwu/CACPP/tree/master), you can find [train_and_test_AntiACP_Main.py](https://github.com/yanngfengwu/CACPP/blob/master/model/train_and_test_AntiACP_Main.py) and [train_and_test_AntiACP_Alternate.py](https://github.com/yanngfengwu/CACPP/blob/master/model/train_and_test_AntiACP_Alternate.py) which are separately used on the Main Dataset and the Alternate Dataset. With the whole project, you can run both files directly.

# Dataset

AntiACP2.0_Alternate and AntiACP2.0_Main comes from the server of AntiCP 2.0

The benchmark dataset and reconstructed datasets in paper is included in [CACPP/dataset](https://github.com/yanngfengwu/CACPP/tree/master/dataset)

The construction of Dataset1, Dataset2, Dataset3, Dataset4, Dataset5 is followed:

| Dataset   | Positive Samples for Train                        | Positive Samples for Test                      | Negative Samples for Train                            | Negative Samples for Test                       |
| --------- | ------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| Dataset 1 | Randomly selected 80% peptides of all ACPs (777)  | The rest 20% of the peptides of all ACPs (193) | Randomly selected 80% peptides of all non-ACPs (1440) | The rest 20% peptides of all non-ACPs (360)     |
| Dataset 2 | Randomly  selected 60% peptides of all ACPs (582) | Same as Dataset1 (193)                         | Same as Dataset1 (1440)                               | Same as Dataset1 (360)                          |
| Dataset 3 | Randomly selected 40% peptides of all ACPs (387)  | Same as Dataset1 (193)                         | Same as Dataset1 (1440)                               | Same as Dataset1 (360)                          |
| Dataset 4 | Randomly selected 80% peptides of all ACPs (777)  | The rest 20% peptides of all ACPs (193)        | Negative samples in Alternate Dataset for train(776)  | Negative samples in Main Dataset for test (172) |
| Dataset 5 | Same as Dataset 4 (777)                           | Same as Dataset 4 (193)                        | Negative samples in Main Dataset for train (689)      | Same as Dataset 4 (172)                         |

## Usage

```python
python train_and_test_AntiACP_Alternate.py  # model on the Alternate Dataset
python train_and_test_AntiACP_Main.py  # model on the Main Dataset
```

