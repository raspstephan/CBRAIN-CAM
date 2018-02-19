#!/usr/bin/env bash
PROC_ROOT=/export/home/srasp/repositories/CBRAIN-Keras-Diagnostics/data_processing/

python $PROC_ROOT/preprocess_aqua.py \
--config_file $PROC_ROOT/config/pure_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref purecrm_ess_train_fullyear &&
python $PROC_ROOT/preprocess_aqua.py \
--config_file $PROC_ROOT/config/pure_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0001-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref purecrm_ess_valid_fullyear \
--ext_norm /beegfs/DATA/pritchard/srasp/preprocessed_data/purecrm_ess_train_fullyear_norm.nc &&
python $PROC_ROOT/shuffle_ds.py \
--pref /beegfs/DATA/pritchard/srasp/preprocessed_data/purecrm_ess_train_fullyear