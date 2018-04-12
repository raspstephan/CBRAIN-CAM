#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/full_physics_essentialsv2.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*-0[5-9]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fullphy_essv2_nonorm_train_sample1 &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/full_physics_essentialsv2.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*-2[1-5]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fullphy_essv2_nonorm_valid_sample1 \
--ext_norm /scratch/srasp/preprocessed_data/fullphy_essv2_nonorm_train_sample1_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /scratch/srasp/preprocessed_data/fullphy_essv2_nonorm_train_sample1