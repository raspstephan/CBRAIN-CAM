#!/usr/bin/env bash
REPO=/export/home/srasp/repositories/CBRAIN-CAM/

python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/pure_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref purecrm_essv2_nonorm_train_fullyear &&
python $REPO/cbrain/preprocess_aqua.py \
--config_file $REPO/pp_config/pure_crm_essentials.yml \
--in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ \
--aqua_names='AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0001-*' \
--out_dir /scratch/srasp/preprocessed_data/ \
--out_pref purecrm_essv2_nonorm_valid_fullyear \
--ext_norm /scratch/srasp/preprocessed_data/purecrm_essv2_nonorm_train_fullyear_norm.nc &&
python $REPO/cbrain/shuffle_ds.py \
--pref /scratch/srasp/preprocessed_data/purecrm_essv2_nonorm_train_fullyear