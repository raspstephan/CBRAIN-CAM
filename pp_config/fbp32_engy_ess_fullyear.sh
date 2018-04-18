#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /scratch/srasp/fluxbypass_aqua32/ \
--aqua_names='*.h1.0000-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref fbp32_engy_ess_train_fullyear &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /scratch/srasp/fluxbypass_aqua32/ \
--aqua_names='*.h1.0000-*-2[1-5]-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref fbp32_engy_ess_valid_fullyear \
--ext_norm /beegfs/DATA/pritchard/srasp/preprocessed_data/fbp32_engy_ess_train_fullyear_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /beegfs/DATA/pritchard/srasp/preprocessed_data/fbp32_engy_ess_train_fullyear
