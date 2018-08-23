#!/usr/bin/env bash
REPO=/home/s/S.Rasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/sp8fbp_4k/ \
--aqua_names='*.h2.0001-*-*-*' \
--out_dir /local/S.Rasp/preprocessed_data/ \
--out_pref fbp_engy_ess_4k_train_fullyear \
--ext_norm mo_salah&&
python $REPO/cbrain/shuffle_ds.py \
--pref /local/S.Rasp/preprocessed_data/fbp_engy_ess_4k_train_fullyear
