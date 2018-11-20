from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

'''
############################################################################################
# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets
model = GraphConvModel(
    len(tox21_tasks), batch_size=50, mode='classification')
# Set nb_epoch=10 for better results.
model.fit(train_dataset, nb_epoch=1)
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])

'''

###########################################################################################
# Load HIV dataset
hiv_tasks, hiv_datasets, transformers = dc.molnet.load_hiv(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = hiv_datasets
model = GraphConvModel(
    len(hiv_tasks), batch_size=70, mode='classification')
# Set nb_epoch=10 for better results.
model.fit(train_dataset, nb_epoch=1)
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])

'''
############################################################################################
# Load SAMPL(FreeSolv) dataset
SAMPL_tasks, SAMPL_datasets, transformers = dc.molnet.load_sampl(
    featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = SAMPL_datasets

# Batch size of models
batch_size = 50
model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

#############################################################################
# Define metric for eavluating the model by using pearson_r2_socre
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Batch size of models
batch_size = 50
model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores (Pearson_R2)")
print(train_scores)

print("Validation scores (Pearson_R2)")
print(valid_scores)

#############################################################################
##  the following is used to evaluate the model by mean_squared_error
# Define metric
metric = dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)

## Batch size of models
batch_size = 50
model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores (MSE)")
print(train_scores)

print("Validation scores (MSE)")
print(valid_scores)
'''
