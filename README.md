# Dual Sparse Attention Network For Session based Recommendation

This code is used to reproduce the main experiment of our paper. We will open source code of baselines after cleaning the code.

## Requirements

+ Python 3.6.8
+ Pytorch 1.2.0
+ entmax (pip install entmax)
+ Jupyter Notebbok

## Datasets

+ DIGINETICA: http://cikm2016.cs.iupui.edu/cikm-cup or https://competitions.codalab.org/competitions/11161
+ RETAILROCKET: https://www.kaggle.com/retailrocket/ecommerce-dataset 

## Code
+ preprocess_rr: for RETAILROCKET dataset to generate session.
+ Preprocess: generate train and test set(for RETAILROCKET dataset, you need run `preprocess_rr .py` first)
+ Metric: HR and MRR
+ DualAdaptiveTrain: the model of DN dataset
+ DualAdaRR3: the model of RR dataset

## BestModel
This folder contains the model that we have trained. Loading this model could directly check results.

## Baselines
This folder contains all the baselines we compared in the paper. For SKNN, STAN,STAMP,Bert4Rec and CoSAN we implement them by ourselves referring to the original paper and open source implementation. For GRU4Rec and SR-GNN, we use the author's source code. For FPMC, we use the open source implementation.