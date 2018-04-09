#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-Keras-Diagnostics/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_cons.yml \
--in_dir /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0000-*-0[5-9]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_cons_train_sample1 &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_cons.yml \
--in_dir /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0000-*-2[1-5]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_cons_valid_sample1 \
--ext_norm /scratch/srasp/preprocessed_data/fbp_engy_cons_train_sample1_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /scratch/srasp/preprocessed_data/fbp_engy_cons_train_sample1