# Connectome-based-predictive-modeling---PET-score

Predictive models created to predict Psychometric Entrance Test (PET) from resting state fMRI data

There is a great scientific interest in understanding individual differences in cognitive traits and relating them to brain connectivity. In previous research, scientists have been able to predict various cognitive measurements from brain connectivity patterns extracted from functional magnetic resonance imaging (fMRI) scans. However, these measurements are often evaluated under laboratory-constrained settings rather than in real-life environment. Here, we predict Psychometric Entrance Test (PET) scores, that are correlated with academic success, from resting-state functional connectivity using machine learning models. 
We used resting-state fMRI scans acquired at Tel Aviv University’s Strauss Center for Computational Neuroimaging, to predict subjects’ PET scores and additional demographic information.


in this repository you can find 4 files:

all_models_322_subj: a jupyter notebook with all of the predictive models showcased in this study, besides deep neural network (DNN), applied to all participants in order to predict their general PET scores.

all_models_158_subj: a jupyter notebook with all of the predictive models showcased in this study, besides deep neural network (DNN), applied to 158 of the participants with the relevant data, in order to predict their math and verbal PET scores.

dnn_322: basic deep neural network applied to all participants in order to predict their general PET scores.

dnn_158: basic deep neural network applied to 158 of the participants with the relevant data, in order to predict their math and verbal PET scores.
