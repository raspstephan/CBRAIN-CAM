#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /beegfs/DATA/pritchard/srasp/sp8fbp_4k_fullout/ \
--aqua_names='*.h1.0000-01-0[1-6]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_ess_valid_4k \
--ext_norm /scratch/srasp/preprocessed_data/fbp_engy_ess_train_fullyear_norm.nc 