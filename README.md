# Signature_Detection_Analysis
Authentication of handwritten signatures using digital image processing and neural networks.

## Dataset Used : Signature verification data
The dataset used was gotten from the ICDAR 2009 Signature Verification Competition (SigComp2009).
Link to the data: http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2009_Signature_Verification_Competition_(SigComp2009)  

## In input_data folder: training and testing data for model
Used BHSig260 folder BHSig260/Hindi as well as BHSig260/Bengali signature folder data for training and testing.

## Model.py & Tensorflow_details.py
Contains code related to signet model used for training.
The Implementation of SigNet in carried out, it's a revolutionary siamese architecture that uses CNNs to learn to differentiate between genuine and forged signatures on BHSig260 dataset.

## Preprocessing.py
Extraction of all the images in data folder directory into orig_groups & forg_group.

## Run.py
Uses all the above python files and includes step by step model training.

## Steps to execute:

1. pip3 install -r requirements.txt
2. python3 run.py


