#!/bin/csh
module load nco
set rootdir="/beegfs/DATA/pritchard/srasp"
set varname='PRECT'
set savedir="/beegfs/DATA/pritchard/srasp/wkdata"
mkdir -p scratch
mkdir -p $savedir
foreach EXP ( 'nnatmonly_fbp8_D040_andkua_nofix_betafix35' ) 
  mkdir -p figs/$EXP
  setenv FILENAME "./scratch/${varname}_rcat_${EXP}.nc"
  if ( ! -e $FILENAME ) then
    # time concatenate variable of interest across raw output files:
    # Note Stephan saved 6-hourly so compute the daily mean as follows:
    # nb limiting to the first year, maximum overlap between sims.
    ncrcat -D 2 -v $varname -d time,1460,8756,4 $rootdir/$EXP/*.cam2.h1.000[12345]-[01]*.nc -o scratch/scratch0_${varname}_${EXP}.nc &
    ncrcat -D 2 -v $varname -d time,1461,8757,4 $rootdir/$EXP/*.cam2.h1.000[12345]-[01]*.nc -o scratch/scratch1_${varname}_${EXP}.nc &
    ncrcat -D 2 -v $varname -d time,1462,8758,4 $rootdir/$EXP/*.cam2.h1.000[12345]-[01]*.nc -o scratch/scratch2_${varname}_${EXP}.nc &
    ncrcat -D 2 -v $varname -d time,1463,8759,4 $rootdir/$EXP/*.cam2.h1.000[12345]-[01]*.nc -o scratch/scratch3_${varname}_${EXP}.nc &
    # (daily means avoid distraction from diurnal cycle; usual starting place for eq. wave analysis)
    wait
    ncea -D 2 scratch/scratch[0123]_${varname}_${EXP}.nc -o $FILENAME
    rm scratch/scratch[0123]_${varname}_${EXP}.nc
  endif
  # Make the Wheeler-Kiladis spectra for this experiment:
  cat wkspectra.ncl | sed "s@VARNAME@${varname}@g" | sed "s@EXP@${EXP}@g" > tmp.$varname.ncl
  ncl < tmp.$varname.ncl
  rm tmp.$varname.ncl
  mv SpaceTime.PRECT.nc $savedir/SpaceTime.PRECT.${EXP}.nc
end

# For my favorite subfigures (equatorially symmetric total, sig-to-noise)
# use ImageMagick to intercompare all the experiments:
# mkdir -p figs/compare
# foreach figprefix ( "Fig1.Sym" "Fig3.Sym" )
#   set filelist = `find . -name "$figprefix*"`
#   montage $filelist -geometry +0+0 figs/compare/compare_$figprefix.jpeg
# end
