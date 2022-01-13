# multipitch_architectures

This is a pytorch code repository accompanying the following paper:  

> Christof Weiß and Geoffroy Peeters  
> _Deep-Learning Architectures for Multi-Pitch Estimation: Towards Reliable Evaluation_  
> submitted to IEEE/ACM Transactions on Audio, Speech &amp; Language Processing, 2022

This repository only contains exemplary code and pre-trained models for most of the paper's experiments as well as some individual examples. All datasets used in the paper are publicly available (at least partially):
* [MusicNet](https://zenodo.org/record/5120004)
* [Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ)
* [TRIOS](http://c4dm.eecs.qmul.ac.uk/rdr/handle/123456789/27)
* [Bach10](http://www2.ece.rochester.edu/projects/air/resource.html)
* [PHENICX-Anechoic](https://www.upf.edu/web/mtg/phenicx-anechoic)
* [Choral Singing Dataset](https://zenodo.org/record/2649950#.Yd_3hWhKhaQ)
  
For details and references, please see the paper.
  
In addition, we provide information on version duplicates in MusicNet (_MusicNet_stats.md_) and detailed information on the different training-test splits used in our experiments (as _JSON_ and _Markdown_ files in folder _dataset_splits_).

# Feature extraction and prediction (Jupyter notebooks)

In this top folder, two Jupyter notebooks (_01_precompute_features_ and _02_predict_with_pretrained_model_) demonstrate how to preprocess audio files for running our models and how to load a pretrained model for predicting pitches.

# Experiments from the paper (Python scripts)

In the _experiments_ folder, all experimental scripts as well as the log files (subfolder _logs_) and the filewise results (subfolder _results_filewise_) can be found. The folder _models_pretrained_ contains pre-trained models for the main experiments. The subfolder _predictions_ contains exemplary model predictions for two of the experiments. Plese note that re-training requires a GPU as well as the pre-processed training data (see the notebook _01_precompute_features_ for an example). Any script must be started from the repository top folder path in order to get the relative paths working correctly.

The experiment files' names relate to the paper's results in the following way:
  
  
  
## Exp1_SectionIV-B
Experiments from __Section IV.B (Table II / Fig. 4) - Model Architectures and Sizes__. Suffix __ _rerun_ denotes additional training/test runs of a model.

### (a) CNN (simple)
* __CNN:XS__&emsp;  _exp126a_musicnet_cnn_basic_
* __CNN:S__&emsp; _exp126b_musicnet_cnn_wide_
* __CNN:M__&emsp; _exp126c_musicnet_cnn_verywide_
* __CNN:L__&emsp; _exp126d_musicnet_cnn_extremelywide_

### (b) DCNN (deep)
* __DCNN:S__&emsp; _exp127a_musicnet_cnn_deepbasic_
* __DCNN:M__&emsp; _exp127b_musicnet_cnn_deepwide_
* __DCNN:L__&emsp; _exp127c_musicnet_cnn_deepverywide_

### (c) DRCNN (deep residual)
* __DRCNN:S__&emsp; _exp128a_musicnet_cnn_deepresnetbasic_
* __DRCNN:M__&emsp; _exp128b_musicnet_cnn_deepresnetwide_
* __DRCNN:L__&emsp; _exp128c_musicnet_cnn_deepresnetverywide_
* __&emsp; &mdash; &emsp;__ &emsp; _exp128c_musicnet_cnn_deepresnetverywide_rerun1_
* __&emsp; &mdash; &emsp;__ &emsp; _exp128c_musicnet_cnn_deepresnetverywide_rerun2_

### (d) Unet
* __Unet:S__&emsp; _exp160d2_musicnet_unet_large_bugfix_
* __Unet:M__&emsp; _exp160g_musicnet_unet_medium_bugfix_
* __&emsp; &mdash; &emsp;__&emsp; _exp160g_musicnet_unet_medium_bugfix_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp160g_musicnet_unet_medium_bugfix_rerun2_
* __Unet:L__&emsp; _exp160e3_musicnet_unet_verylarge_bugfix_scaled_
* __&emsp; &mdash; &emsp;__&emsp; _exp160e3_musicnet_unet_verylarge_bugfix_scaled_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp160e3_musicnet_unet_verylarge_bugfix_scaled_rerun2_
* __Unet:XL__&emsp; _exp160f_musicnet_unet_veryverylarge_
* __&emsp; &mdash; &emsp;__&emsp; _exp160f_musicnet_unet_veryverylarge_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp160f_musicnet_unet_veryverylarge_rerun2_

### (e) SAUnet (self-attention at bottleneck)
* __SAUnet:M__&emsp; _exp180b_musicnet_unet_verylarge_doubleselfattn_
* __SAUnet:L__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_rerun2_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_rerun3_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_rerun4_
* __SAUnet:XL__&emsp; _exp180e_musicnet_unet_insanelylarge_doubleselfattn_
* __&emsp; &mdash; &emsp;__&emsp; _exp180e_musicnet_unet_insanelylarge_doubleselfattn_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp180e_musicnet_unet_insanelylarge_doubleselfattn_rerun2_
* __SAUnet:XXL__&emsp; _exp180f_musicnet_unet_intermedlarge_doubleselfattn_
* __&emsp; &mdash; &emsp;__&emsp; _exp180f_musicnet_unet_intermedlarge_doubleselfattn_rerun_

### (f) SAUSnet (self-attention also at lowest skip connection)
* __SAUSnet:M__&emsp; _exp181b_musicnet_unet_verylarge_doubleselfattn_twolayers_
* __SAUSnet:L__&emsp; _exp181d_musicnet_unet_verylarge_doubleselfattn_twolayers_
* __SAUSnet:XL__&emsp; _exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_
* __&emsp; &mdash; &emsp;__&emsp; _exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_rerun2_
* __SAUSnet:XXL__&emsp; _exp181e_musicnet_unet_insanelylarge_doubleselfattn_twolayers_

### (g) BLUnet (BiLSTM at bottleneck)
* __BLUnet:M__&emsp; _exp186b_musicnet_unet_verylarge_blstm_
* __BLUnet:L__&emsp; _exp186d_musicnet_unet_extremelylarge_blstm_
* __BLUnet:XXL__&emsp; _exp186e_musicnet_unet_insanelylarge_blstm_

### (h) PUnet (multi-task with degree-of-polyphony estimation)
* __PUnet:M__&emsp; _exp195g_musicnet_unet_extremelylarge_polyphony_softmax_
* __PUnet:L__&emsp; _exp195e3_musicnet_unet_extremelylarge_polyphony_softmax_
* __PUnet:XL__&emsp; _exp195f_musicnet_unet_extremelylarge_polyphony_softmax_
* __&emsp; &mdash; &emsp;__&emsp; _exp195f_musicnet_unet_extremelylarge_polyphony_softmax_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp195f_musicnet_unet_extremelylarge_polyphony_softmax_rerun2_
  
  
  
## Exp2_SectionIV-C
Experiments from __Section IV.C (Table IV) - Model Generalization (more training samples, other testsets)__. Suffix __ _rerun_ denotes additional training/test runs of a model.

### (a) Test set MuN-10a (more training samples)
* __Unet:XL__&emsp; _exp160f_musicnet_unet_veryverylarge_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _exp160f_musicnet_unet_veryverylarge_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp160f_musicnet_unet_veryverylarge_moresamples_rerun2_
* __SAUnet:L__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun2_
* __SAUSnet:XL__&emsp; _exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_moresamples_
* __PUnet:XL__&emsp; _exp195f_musicnet_unet_extremelylarge_polyphony_softmax_moresamples_

### (b) Test set MuN-10 (original)
* __Unet:XL__&emsp; _RETRAIN_exp160f_musicnet_unet_veryverylarge_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp160f_musicnet_unet_veryverylarge_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp160f_musicnet_unet_veryverylarge_moresamples_rerun2_
* __SAUnet:L__&emsp; _RETRAIN_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun2_
* __SAUSnet:XL__&emsp; _RETRAIN_exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN_exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_moresamples_rerun2_
* __PUnet:XL__&emsp; _RETRAIN_exp195f_musicnet_unet_extremelylarge_polyphony_softmax_

### (c) Test set MuN-3 (90s) 
* see models from __(a) Test set MuN-10a__

### (d) Test set MuN-10b (slow movements)
* __SAUnet:L__&emsp; _RETRAIN2_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_

### (e) Test set MuN-10c (fast movements)
* __SAUnet:L__&emsp; _RETRAIN3_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_

### (f) Test set MuN-10full (all movements of the ten work cycles)
* __CNN:M__&emsp; _RETRAIN4_exp127c_musicnet_cnn_verywide_moresamples_
* __DRCNN:L__&emsp; _RETRAIN4_exp128c_musicnet_cnn_deepresnetwide_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp128c_musicnet_cnn_deepresnetwide_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp128c_musicnet_cnn_deepresnetwide_moresamples_rerun2_
* __Unet:M__&emsp; _RETRAIN4_exp160f_musicnet_unet_veryverylarge_moresamples_
* __Unet:XL__&emsp; _RETRAIN4_exp160g_musicnet_unet_medium_moresamples_
* __SAUnet:L__&emsp; _RETRAIN4_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp180d_musicnet_unet_extremelylarge_doubleselfattn_moresamples_rerun2_
* __SAUSnet:XL__&emsp; _RETRAIN4_exp181f_musicnet_unet_intermedlarge_doubleselfattn_twolayers_moresamples_
* __BLUnet:L__&emsp; _RETRAIN4_exp186d_musicnet_unet_extremelylarge_blstm_moresamples_
* __PUnet:XL__&emsp; _RETRAIN4_exp195f_musicnet_unet_extremelylarge_polyphony_softmax_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp195f_musicnet_unet_extremelylarge_polyphony_softmax_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _RETRAIN4_exp195f_musicnet_unet_extremelylarge_polyphony_softmax_rerun2_
  
  
  
## Exp3_SectionIV-D
Experiments from __Section IV.D (Fig. 6) - Cross-Version Study on Schubert Winterreise__.

### CNN:M
* __Version split:__&emsp; _exp200a_schubert_versionsplit_cnn_verywide_
* __Song split:__&emsp; _exp200b_schubert_songsplit_cnn_verywide_
* __Neither split:__&emsp; _exp200c_schubert_neithersplit_cnn_verywide_

### SAUnet:L
* __Version split:__&emsp; _exp201a_schubert_versionsplit_unet_extremelylarge_doubleselfattn_
* __Song split:__&emsp; _exp201b_schubert_songsplit_unet_extremelylarge_doubleselfattn_
* __Neither split:__&emsp; _exp201c_schubert_neithersplit_unet_extremelylarge_doubleselfattn_
  
  
  
## Exp4_SectionIV-E
Experiments from __Section IV.E (Fig. 7) - Cross-Dataset Study on Big Mix Dataset__, compiled from all source datasets. Suffix __ _rerun_ denotes additional training/test runs of a model.

* __CNN:M__&emsp; _exp216c_bigmix_cnn_verywide_
* __&emsp; &mdash; &emsp;__&emsp; _exp216c_bigmix_cnn_verywide_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp216c_bigmix_cnn_verywide_rerun2_
* __DRCNN:L__&emsp; _exp214c_bigmix_cnn_deepresnetwide_
* __&emsp; &mdash; &emsp;__&emsp; _exp214c_bigmix_cnn_deepresnetwide_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp214c_bigmix_cnn_deepresnetwide_rerun2_
* __Unet:M__&emsp; _exp213g_bigmix_unet_medium_
* __&emsp; &mdash; &emsp;__&emsp; _exp213g_bigmix_unet_medium_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp213g_bigmix_unet_medium_rerun2_
* __Unet:XL__&emsp; _exp212f_bigmix_unet_veryverylarge_
* __&emsp; &mdash; &emsp;__&emsp; _exp212f_bigmix_unet_veryverylarge_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp212f_bigmix_unet_veryverylarge_rerun2_
* __SAUnet:L__&emsp; _exp210d_bigmix_unet_extremelylarge_doubleselfattn_
* __&emsp; &mdash; &emsp;__&emsp; _exp210d_bigmix_unet_extremelylarge_doubleselfattn_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp210d_bigmix_unet_extremelylarge_doubleselfattn_rerun2_
* __SAUSnet:XL__&emsp; _exp211f_bigmix_unet_intermedlarge_doubleselfattn_twolayers_
* __&emsp; &mdash; &emsp;__&emsp; _exp211f_bigmix_unet_intermedlarge_doubleselfattn_twolayers_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp211f_bigmix_unet_intermedlarge_doubleselfattn_twolayers_rerun2_
* __BLUnet:L__&emsp; _exp217d_bigmix_unet_extremelylarge_blstm_
* __&emsp; &mdash; &emsp;__&emsp; _exp217d_bigmix_unet_extremelylarge_blstm_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp217d_bigmix_unet_extremelylarge_blstm_rerun2_
* __PUnet:XL__&emsp; _exp215f_bigmix_unet_extremelylarge_polyphony_softmax_
* __&emsp; &mdash; &emsp;__&emsp; _exp215f_bigmix_unet_extremelylarge_polyphony_softmax_rerun1_
* __&emsp; &mdash; &emsp;__&emsp; _exp215f_bigmix_unet_extremelylarge_polyphony_softmax_rerun2_

  
  
  
Run scripts using e.g. the following commands:  
__conda activate multipitch_architectures__  
__export CUDA_VISIBLE_DEVICES=1__  
__python experiments/Exp1_SectionIV-B/exp126a_musicnet_cnn_basic.py__
