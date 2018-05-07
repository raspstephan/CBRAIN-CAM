#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /beegfs/DATA/pritchard/srasp/sp8fbp_4k/ /scratch/srasp/fluxbypass_aqua/ \
--aqua_names='*.h2.0001-*-[14-31]-*' '*.h1.0001-*-0[14-31]-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref fbp_engy_ess_train_ref_4k_fullyear \
--ext_norm mo_salah &&
python $REPO/cbrain/shuffle_ds.py \
--pref /scratch/srasp/preprocessed_data/fbp_engy_ess_train_ref_4k_fullyear


