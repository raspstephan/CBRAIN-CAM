#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='*.h1.0000-*-1[7-9]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_ess_ref_valid \
--ext_norm mo_salah