#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/full_physics_essentialsv2.yml \
--in_dir /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0000-*-0[5-9]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fullphy_fbp_train_sample1 &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/full_physics_essentialsv2.yml \
--in_dir /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0000-*-2[1-5]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fullphy_fbp_valid_sample1 \
--ext_norm /scratch/srasp/preprocessed_data/fullphy_fbp_train_sample1_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /scratch/srasp/preprocessed_data/fullphy_fbp_train_sample1