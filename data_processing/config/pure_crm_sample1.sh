#!/usr/bin/env bash
PROC_ROOT=/export/home/srasp/repositories/CBRAIN-Keras-Diagnostics/data_processing/

python $PROC_ROOT/preprocess_aqua.py \
--config_file $PROC_ROOT/config/pre_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*-0[5-9]-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref purecrm_ess_train_sample1 &&
python $PROC_ROOT/preprocess_aqua.py \
--config_file $PROC_ROOT/config/pre_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*-2[1-5]-*' \
--out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ \
--out_pref purecrm_ess_valid_sample1 \
--ext_norm /beegfs/DATA/pritchard/srasp/preprocessed_data/purecrm_ess_train_sample1_norm.nc &&
python $PROC_ROOT/shuffle_ds.py \
--pref /beegfs/DATA/pritchard/srasp/preprocessed_data/purecrm_ess_train_sample1