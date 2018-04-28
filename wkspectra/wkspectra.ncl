load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"
begin
  filename = getenv ("FILENAME")
  f= addfile (filename, "r")
  latN     = 15
  latS     = -latN       ; symmetric about the equator
  vardata = f->VARNAME(:,{latS:latN},:)

  nDayWin  = 96      ; Wheeler-Kiladis [WK] temporal window length (days)
  nDaySkip = -65      ; time (days) between temporal windows [segments]
                     ; negative means there will be overlapping
                     ; temporal segments

  opt      = True
  opt@pltType = "ps"
  opt@netCDF = True
  opt@Fig_1 = fspan (-17.3,-15.6,15)
  spd      = 1       ; samples per day (6 hour sampling rate)
  diro = "./figs/EXP"
  case = "EXP"
  var = "VARNAME"

  wkSpaceTime (vardata,diro,case,var,latN,spd,nDayWin,nDaySkip,opt)

end

