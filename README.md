# Attentive Reader

Torch implementation of Attentive Reader network from "Teaching Machines to Read and Comprehend" ([Hermann, 2015](https://arxiv.org/pdf/1506.03340v3)). Code uses v1.0 of [bAbI dataset](https://research.fb.com/projects/babi/) with 1k questions per task.

## Prerequisites:
- Python 2.7
- Torch (with nngraph)

## Preprocessing
First, preprocess included data into hdf5 format:
```
python preprocess.py
```
This will create a hdf5 file for each task (total 20 tasks).

To train:
```
th train.lua
```
This will train on task 1 over 250 epochs by default. See `train.lua` (or the paper) for hyperparameters and more training options.
To train on gpu, use `-cuda`.
Full result on the 20 tasks is listed below (numbers are not final for tasks 2 & 3 due to slow training).
NOTE: because of training time these numbers are reported from a single run whereas for bAbI tasks multiple runs are typically performed and the best result is reported.
In addition, the parameters could likely be tuned much better. At the moment, convergence could require > 100 epochs and sometimes a restart is needed to achieve good performance.
 

| Task                        | Accuracy (%) |
|-----------------------------|--------------|
| 01 - Single Supporting Fact | 98.1         |
| 02- Two Supporting Facts    | 33.6         |
| 03 - Three Supporting Facts | 25.5         |
| 04 - Two Arg Relations      | 98.5         |
| 05 - Three Arg Relations    | 97.8         |
| 06 - Yes No Questions       | 55.6         |
| 07 - Counting               | 80.0         |
| 08 - List Sets              | 92.1         |
| 09 - Simple Negation        | 64.3         |
| 10 - Indefinite Knowledge   | 57.2         |
| 11 - Basic Coreference      | 94.4         |
| 12 - Conjunction            | 93.6         |
| 13 - Compound Coreference   | 94.4         |
| 14 - Time Reasoning         | 75.3         |
| 15 - Basic Deduction        | 57.6         |
| 16 - Basic Induction        | 50.4         |
| 17 - Positional Reasoning   | 63.1         |
| 18 - Size Reasoning         | 92.7         |
| 19 - Path Finding           | 11.5         |
| 20 - Agents Motivation      | 98.0         |
| **Average**                 | **71.7**     |
