#!/usr/bin/env bash
REPO=/home/s/S.Rasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /local/S.Rasp/sp8fbp_andkua_nobeta/ \
--aqua_names='sp8fbp_andkua_nobeta.cam2.h2.0000-*-0[5-9]-*' \
--out_dir /local/S.Rasp/preprocessed_data/ \
--out_pref fbp_nobeta_engy_cons_train_sample1 &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/fbp_engy_ess.yml \
--in_dir /local/S.Rasp/sp8fbp_andkua_nobeta/ \
--aqua_names='sp8fbp_andkua_nobeta.cam2.h2.0000-*-2[1-5]-*' \
--out_dir /local/S.Rasp/preprocessed_data/ \
--out_pref fbp_nobeta_engy_cons_valid_sample1 \
--ext_norm /local/S.Rasp/preprocessed_data/fbp_nobeta_engy_cons_train_sample1_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /local/S.Rasp/preprocessed_data/fbp_nobeta_engy_cons_train_sample1
