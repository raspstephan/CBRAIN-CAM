#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /beegfs/DATA/pritchard/srasp/sp8fbp_1k/ \
--aqua_names='*.h2.0000-*-1[7-9]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_ess_1k_valid \
--ext_norm mo_salah