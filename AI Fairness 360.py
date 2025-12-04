#Jupyter Notebook (.ipynb)✅ PART 3 — FAIRNESS AUDIT NOTEBOOK (Python)
#Below is a full Jupyter Notebook–ready script using AI Fairness 360, Pandas, Matplotlib, and the COMPAS dataset.

#📘 Fairness Audit Notebook Code (copy-paste into .ipynb)
# --------------------------------------------
# AI ETHICS ASSIGNMENT – FAIRNESS AUDIT (COMPAS)
# --------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load COMPAS dataset
dataset = CompasDataset()
df = dataset.convert_to_dataframe()[0]

print("Dataset shape:", df.shape)
df.head()

# Protected attribute: race
protected = 'race'
privileged_groups = [{'race': 1}]   # Caucasian
unprivileged_groups = [{'race': 0}] # African-American

# -----------------------------
# Fairness Check Before Training
# -----------------------------
metric = BinaryLabelDatasetMetric(dataset, 
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

print("Disparate Impact:", metric.disparate_impact())

# Train-Test Split
X = df.drop(columns=['two_year_recid'])
y = df['two_year_recid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Convert to AIF360 dataset for fairness metrics
test_dataset = dataset.split([0.7], shuffle=True)[1]

classified_metric = ClassificationMetric(
    test_dataset,
    test_dataset.copy().set_labels(y_pred),
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("Equal Opportunity Difference:", classified_metric.equal_opportunity_difference())
print("Average odds difference:", classified_metric.average_odds_difference())
print("False Positive Rate Difference:", classified_metric.false_positive_rate_difference())

# -----------------------------
# Bias Mitigation – Reweighing
# -----------------------------
RW = Reweighing(unprivileged_groups, privileged_groups)
dataset_transformed = RW.fit_transform(dataset)

print("New weights created by reweighing.")

# Plot weights distribution
plt.hist(dataset_transformed.instance_weights, bins=50)
plt.title("Reweighing Weight Distribution")
plt.xlabel("Weight")
plt.ylabel("Count")
plt.show()
