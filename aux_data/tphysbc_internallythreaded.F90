#include <misc.h>
#include <params.h>
#define PCWDETRAIN
#define RADTIME 900.
#define SP_DIR_NS
#define NOMOMTRANS
#undef CRM3D
subroutine tphysbc_internallythreaded (ztodt,   pblht,   tpert,   in_srfflx_state2d, & 
                    qpert, in_surface_state2d, &
                    snowh,   &
                    qrs,     qrl,  &
                    in_fsns,    in_fsnt,    &
                    in_flns,    in_flnt,  &
                    state,   tend,    &
                    pbuf,    in_prcsnw,  fsds ,   in_landm,   in_landfrac,&
         		    in_ocnfrac, in_icefrac  &
#ifdef CRM
                   ,u_crm, v_crm, w_crm, t_crm, q_crm, qn_crm, qp_crm &
                  ,qrs_crm, qrl_crm, in_rad_buffer, qrs1, qrl1  &
		   ,fsds_crm,fsns_crm,fsntoa_crm,fsutoa_crm  &
		   ,flwds_crm,flns_crm,flut_crm   &!
		   ,fsdsc_crm,fsntoac_crm,flnsc_crm, flutc_crm & 
#endif
                      )
!-----------------------------------------------------------------------
!
! Purpose:
! Tendency physics BEFORE coupling to land, sea, and ice models.
!
! Method:
! Call physics subroutines and compute the following:
!     o cloud calculations (cloud fraction, emissivity, etc.)
!     o radiation calculations
! Pass surface fields for separate surface flux calculations
! Dump appropriate fields to history file.
! 
! Author: CCM1, CMS Contact: J. Truesdale
! 
!-----------------------------------------------------------------------

   use shr_kind_mod,    only: r8 => shr_kind_r8
   use ppgrid
   use pmgrid, only: masterproc
   use phys_grid,       only: get_rlat_all_p, get_rlon_all_p, get_lon_all_p, get_lat_all_p
   use phys_buffer,     only: pbuf_size_max, pbuf_fld, pbuf_old_tim_idx, pbuf_get_fld_idx
   use cldcond,         only: cldcond_tend, cldcond_zmconv_detrain, cldcond_sediment
   use comsrf, only: srfflx_state, surface_state
   use param_cldoptics, only: param_cldoptics_calc
   use physics_types,   only: physics_state, physics_tend, physics_ptend, physics_update, physics_ptend_init
   use diagnostics,     only: diag_dynvar
   use history,         only: outfld, fillvalue
   use physconst,       only: gravit, latvap, cpair, tmelt, cappa, zvir, rair, rga
   use radheat,         only: radheat_net
   use constituents,    only: pcnst, pnats, ppcnst, qmin
   use constituents,    only: dcconnam, cnst_get_ind
   use zm_conv,         only: zm_conv_evap, zm_convr
   use time_manager,    only: is_first_step, get_nstep, get_curr_calday
   use moistconvection, only: cmfmca
   use check_energy,    only: check_energy_chng, check_energy_fix
   use dycore,          only: dycore_is
   use cloudsimulatorparms, only: doisccp
   use cloudsimulator, only: ccm_isccp
   use aerosol_intr, only: aerosol_wet_intr
#ifdef CRM
   use history,         only: outfldcol
   use crmdims,       only: crm_nx, crm_ny, crm_nz
   use buffer,        only: nrad_buffer
   use pkg_cldoptics, only: cldefr, cldems, cldovrlap
   use check_energy,  only: check_energy_timestep_init
   use cloudsimulatorparms
   use cloudsimulator, only: crm_isccp
   use runtime_opts, only: crminitread
   use filenames, only: ncdata
   use ioFileMod,    only: getfil
#ifdef CRMACCEL
   use runtime_opts, only: do_accel, crm_accel_factor
#endif

#endif
   use runtime_opts, only: l_analyses, tau_ps, tau_q, tau_t, tau_u, tau_v, nudge_dse_not_T
   use wv_saturation,  only: aqsat ! pritch

   use analyses,      only: analyses_nudge, t_a, u_a, v_a, q_a, ps_a, s_a
#ifdef QRLDAMP
   use runtime_opts, only: qrldampfac,qrldamp_equatoronly, qrl_dylat, qrl_critlat_deg, qrl_dailymean_interference,qrldamp_freetroponly,qrl_pbot,qrl_ptop,qrl_dp
   use qrl_anncycle, only: accumulate_dailymean_qrl, qrl_interference
#endif

   implicit none

#include <comctl.h>
#include <comlun.h>

   include 'netcdf.inc'

! Arguments
!


   real(r8), intent(in) :: ztodt                          ! 2 delta t (model time increment)
   real(r8), intent(inout) :: pblht(pcols,begchunk:endchunk)                ! Planetary boundary layer height
   real(r8), intent(inout) :: tpert(pcols,begchunk:endchunk)                ! Thermal temperature excess
   type (srfflx_state), intent (inout) :: in_srfflx_state2d(begchunk:endchunk) ! Hey is it really in?
   real(r8), intent(inout) :: qpert(pcols,ppcnst,begchunk:endchunk)         ! Thermal humidity & constituent excess
   real(r8), intent(in) :: snowh(pcols,begchunk:endchunk)                  ! Snow depth (liquid water equivalent)
   real(r8), intent(inout) :: qrs(pcols,pver,begchunk:endchunk)            ! Shortwave heating rate
   real(r8), intent(inout) :: qrl(pcols,pver,begchunk:endchunk)            ! Longwave  heating rate
   type (surface_state), intent (inout) :: in_surface_state2d(begchunk:endchunk)
 
   real(r8), intent(inout) :: in_fsns(pcols,begchunk:endchunk)                   ! Surface solar absorbed flux
   real(r8), intent(inout) :: in_fsnt(pcols,begchunk:endchunk)                   ! Net column abs solar flux at model top
   real(r8), intent(inout) :: in_flns(pcols,begchunk:endchunk)                   ! Srf longwave cooling (up-down) flux
   real(r8), intent(inout) :: in_flnt(pcols,begchunk:endchunk)                   ! Net outgoing lw flux at model top

   type(physics_state), intent(inout) :: state(begchunk:endchunk)
   type(physics_tend ), intent(inout) :: tend(begchunk:endchunk)

   type(pbuf_fld),      intent(inout), dimension(pbuf_size_max) :: pbuf

    real(r8), intent(out) :: in_prcsnw(pcols,begchunk:endchunk)                 ! snowfall rate (precsl + precsc)
   real(r8), intent(out) :: fsds(pcols,begchunk:endchunk)                   ! Surface solar down flux
   real(r8), intent(in) :: in_landm(pcols,begchunk:endchunk)                   ! land fraction ramp
   real(r8), intent(in) :: in_landfrac(pcols,begchunk:endchunk)                ! land fraction
#ifdef CRM
   real(r8), intent(inout) :: u_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: v_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: w_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: t_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: q_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: qn_crm  (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: qp_crm (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: qrs_crm(pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: qrl_crm(pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8), intent(inout) :: in_rad_buffer(pcols,nrad_buffer,begchunk:endchunk)
   real(r8), intent(inout) :: qrs1(pcols,pver,begchunk:endchunk)
   real(r8), intent(inout) :: qrl1(pcols,pver,begchunk:endchunk)
   real(r8), intent(inout) :: fsds_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Flux Shortwave Downwelling Surface
   real(r8), intent(inout) :: fsns_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Surface solar absorbed flux
   real(r8), intent(inout) :: fsntoa_crm(pcols,crm_nx,crm_ny,begchunk:endchunk) ! Net column abs solar flux at model top
   real(r8), intent(inout) :: fsutoa_crm(pcols,crm_nx,crm_ny,begchunk:endchunk) ! Flux Shortwave Upwelling TOA
   real(r8), intent(inout) :: fsntoac_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)! Clear sky total column abs solar flux
   real(r8), intent(inout) :: fsdsc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)  ! Clear sky downard solar flux surface 
   real(r8), intent(inout) :: flwds_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)  ! Surface longwave down flux
   real(r8), intent(inout) :: flns_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Srf longwave cooling (up-down) flux
   real(r8), intent(inout) :: flut_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Outgoing lw flux at model top
   real(r8), intent(inout) :: flutc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)  ! Clear sky outgoing lw flux at model top
   real(r8), intent(inout) :: flnsc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)  ! Clear sky lw flux at srf (up-down)   
#endif
     
   ! Newly local variables (no longer arguments b/c subsets of srfflx_state etc.)   
   ! (many of these were previously tphysbc input args, subsetted to chunks here though.)
   real(r8) :: ts(pcols,begchunk:endchunk)                      ! surface temperature
   real(r8) :: sst(pcols,begchunk:endchunk)                     ! sea surface temperature






   real(r8) :: asdir(pcols,begchunk:endchunk)                  ! Albedo: shortwave, direct
   real(r8) :: asdif(pcols,begchunk:endchunk)                  ! Albedo: shortwave, diffuse
   real(r8) :: aldir(pcols,begchunk:endchunk)                  ! Albedo: longwave, direct
   real(r8) :: aldif(pcols,begchunk:endchunk)                  ! Albedo: longwave, diffuse
   

   ! Now all parts of in_surface_state2d:
   real(r8) :: flwds(pcols,begchunk:endchunk)               ! Surface longwave down flux  


  
   real(r8) :: lwup(pcols,begchunk:endchunk)                    ! Surface longwave up flux
   real(r8) :: srfrad(pcols,begchunk:endchunk)                 ! Net surface radiative flux (watts/m**2)
   real(r8) :: sols(pcols,begchunk:endchunk)                   ! Direct beam solar rad. onto srf (sw)
   real(r8) :: soll(pcols,begchunk:endchunk)                   ! Direct beam solar rad. onto srf (lw)
   real(r8) :: solsd(pcols,begchunk:endchunk)                  ! Diffuse solar radiation onto srf (sw)
   real(r8) :: solld(pcols,begchunk:endchunk)                  ! Diffuse solar radiation onto srf (lw)
   
   real(r8) :: precl(pcols,begchunk:endchunk)                  ! Large-scale precipitation rate
   real(r8) :: precc(pcols,begchunk:endchunk)                  ! Convective-scale preciptn rate
   real(r8) :: precsl(pcols,begchunk:endchunk)                 ! L.S. snowfall rate
   real(r8) :: precsc(pcols,begchunk:endchunk)                 ! C.S. snowfall rate
   


   real(r8) :: in_ocnfrac(pcols,begchunk:endchunk)                ! land fraction
   real(r8) :: in_icefrac(pcols,begchunk:endchunk)                ! land fraction
   
   ! should now be just part of in_srfflx_state2d existing argument:
   real(r8)    :: taux(pcols,begchunk:endchunk)   ! surface stress (zonal) (N/m2)
   real(r8)    :: tauy(pcols,begchunk:endchunk)   ! surface stress (zonal) (N/m2)
   real(r8)    :: shf(pcols,begchunk:endchunk)   ! surface sensible heat flux (W/m2)
   real(r8)    :: lhf(pcols,begchunk:endchunk)   ! surface latent heat flux (W/m2) 



!
!---------------------------Local workspace-----------------------------
!


   real(r8) :: rhdfda(pcols,pver,begchunk:endchunk)            ! dRh/dcloud, old 
   real(r8) :: rhu00 (pcols,pver,begchunk:endchunk)            ! Rh threshold for cloud, old

   type(physics_ptend)   :: ptend(begchunk:endchunk)                 ! indivdual parameterization tendencies

   integer :: nstep                         ! current timestep number
   integer      lat(pcols,begchunk:endchunk)                   ! current latitudes(indices)
   integer      lon(pcols,begchunk:endchunk)                   ! current longtitudes(indices)

   real(r8) :: calday                        ! current calendar day
   real(r8) :: clat(pcols,begchunk:endchunk)                   ! current latitudes(radians)
   real(r8) :: clon(pcols,begchunk:endchunk)                   ! current longitudes(radians)




   real(r8) :: zdu(pcols,pver,begchunk:endchunk)               ! detraining mass flux from deep convection
   real(r8) :: ftem(pcols,pver,begchunk:endchunk)              ! Temporary workspace for outfld variables
   real(r8) :: zmrprd(pcols,pver,begchunk:endchunk)            ! rain production in ZM convection

   real(r8) :: cmfmc(pcols,pverp,begchunk:endchunk)            ! Convective mass flux--m sub c
   real(r8) :: cmfsl(pcols,pver,begchunk:endchunk)             ! Moist convection lw stat energy flux
   real(r8) :: cmflq(pcols,pver,begchunk:endchunk)             ! Moist convection total water flux
   real(r8) :: dtcond(pcols,pver,begchunk:endchunk)            ! dT/dt due to moist processes
   real(r8) :: dqcond(pcols,pver,ppcnst,begchunk:endchunk)     ! dq/dt due to moist processes

   real(r8) cldst(pcols,pver,begchunk:endchunk)
   real(r8) cltot(pcols,begchunk:endchunk)                      ! Diagnostic total cloud cover
   real(r8) cllow(pcols,begchunk:endchunk)                      !       "     low  cloud cover
   real(r8) clmed(pcols,begchunk:endchunk)                      !       "     mid  cloud cover
   real(r8) clhgh(pcols,begchunk:endchunk)                      !       "     hgh  cloud cover
   real(r8) cmfcme(pcols,pver,begchunk:endchunk)                ! cmf condensation - evaporation
   real(r8) cmfdqr2(pcols,pver,begchunk:endchunk)               ! dq/dt due to moist convective rainout
   real(r8) cmfmc2(pcols,pverp,begchunk:endchunk)               ! Moist convection cloud mass flux
   real(r8) cmfsl2(pcols,pver,begchunk:endchunk)                ! Moist convection lw stat energy flux
   real(r8) cmflq2(pcols,pver,begchunk:endchunk)                ! Moist convection total water flux

   real(r8) cnt(pcols,begchunk:endchunk)                        ! Top level of convective activity
   real(r8) cnb(pcols,begchunk:endchunk)                        ! Lowest level of convective activity
   real(r8) cnt2(pcols,begchunk:endchunk)                       ! Top level of convective activity
   real(r8) cnb2(pcols,begchunk:endchunk)                       ! Bottom level of convective activity
   real(r8) concld(pcols,pver,begchunk:endchunk)             
   real(r8) coszrs(pcols,begchunk:endchunk)                     ! Cosine solar zenith angle
   real(r8) dlf(pcols,pver,begchunk:endchunk)                   ! Detraining cld H20 from convection
   real(r8) pflx(pcols,pverp,begchunk:endchunk)                 ! Conv rain flux thru out btm of lev
   real(r8) prect(pcols,begchunk:endchunk)                      ! total (conv+large scale) precip rate
   real(r8) dlf2(pcols,pver,begchunk:endchunk)                   ! dq/dt due to rainout terms
   real(r8) qpert2(pcols,ppcnst,begchunk:endchunk)              ! Perturbation q
   real(r8) rtdt                              ! 1./ztodt
   real(r8) tpert2(pcols,begchunk:endchunk)                     ! Perturbation T
   real(r8) pmxrgn(pcols,pverp,begchunk:endchunk)               ! Maximum values of pressure for each


!                                             !    maximally overlapped region.
!                                             !    0->pmxrgn(i,1) is range of pressure for
!                                             !    1st region,pmxrgn(i,1)->pmxrgn(i,2) for
!                                             !    2nd region, etc
   integer c,lchnk                                  ! chunk identifier
   integer ncol                              ! number of atmospheric columns 

   integer nmxrgn(pcols,begchunk:endchunk)                      ! Number of maximally overlapped regions
   integer  i,k,m                             ! Longitude, level, constituent indices
   integer :: ixcldice, ixcldliq              ! constituent indices for cloud liquid and ice water.
                                           
!  real(r8) engt                              ! Thermal   energy integral
!  real(r8) engk                              ! Kinetic   energy integral
!  real(r8) engp                              ! Potential energy integral
   real(r8) rel(pcols,pver,begchunk:endchunk)                   ! Liquid cloud particle effective radius
   real(r8) rei(pcols,pver,begchunk:endchunk)                   ! Ice effective drop size (microns)
   real(r8) emis(pcols,pver,begchunk:endchunk)                  ! Cloud longwave emissivity
   real(r8) clc(pcols,begchunk:endchunk)                        ! Total convective cloud (cloud scheme)

   real(r8) :: cicewp(pcols,pver,begchunk:endchunk)             ! in-cloud cloud ice water path
   real(r8) :: cliqwp(pcols,pver,begchunk:endchunk)             ! in-cloud cloud liquid water path
!
   real(r8) dellow(pcols,begchunk:endchunk)                     ! delta p for bottom three levels of model
   real(r8) tavg(pcols,begchunk:endchunk)                       ! mass weighted average temperature for 

! physics buffer fields to compute tendencies for cloud condensation package
   integer itim, ifld
   real(r8), pointer, dimension(:,:) :: qcwat, tcwat, lcwat, cld
! physics buffer fields for total energy and mass adjustment
   real(r8), pointer, dimension(:  ) :: teout
   real(r8), pointer, dimension(:,:) :: qini
   real(r8), pointer, dimension(:,:) :: tini

#ifdef CRM
   real(r8) spcld (pcols,pver,begchunk:endchunk)
#endif
!                                          
! Used for OUTFLD only                     
!                                          



   real(r8) icwmr1(pcols,pver,begchunk:endchunk)                ! in cloud water mixing ration for zhang scheme
   real(r8) icwmr2(pcols,pver,begchunk:endchunk)                ! in cloud water mixing ration for hack scheme
   real(r8) fracis(pcols,pver,ppcnst,begchunk:endchunk)         ! fraction of transported species that are insoluble
   real(r8) timestep(pcols,begchunk:endchunk)
!
!     Variables for doing deep convective transport outside of zm_convr
!

   real(r8) mu2(pcols,pver,begchunk:endchunk)
   real(r8) eu2(pcols,pver,begchunk:endchunk)
   real(r8) du2(pcols,pver,begchunk:endchunk)
   real(r8) md2(pcols,pver,begchunk:endchunk)
   real(r8) ed2(pcols,pver,begchunk:endchunk)
   real(r8) dp(pcols,pver,begchunk:endchunk)
   real(r8) dsubcld(pcols,begchunk:endchunk)
   real(r8) conicw(pcols,pver,begchunk:endchunk)
   real(r8) cmfdqrt(pcols,pver,begchunk:endchunk)               ! dq/dt due to moist convective rainout



! stratiform precipitation variables
   real(r8) :: prec_pcw(pcols,begchunk:endchunk)                ! total precip from prognostic cloud scheme
   real(r8) :: snow_pcw(pcols,begchunk:endchunk)                ! snow from prognostic cloud scheme
   real(r8) :: prec_sed(pcols,begchunk:endchunk)                ! total precip from cloud sedimentation
   real(r8) :: snow_sed(pcols,begchunk:endchunk)                ! snow from cloud ice sedimentation

! convective precipitation variables
   real(r8) :: prec_zmc(pcols,begchunk:endchunk)                ! total precipitation from ZM convection
   real(r8) :: snow_zmc(pcols,begchunk:endchunk)                ! snow from ZM convection
   real(r8) :: prec_cmf(pcols,begchunk:endchunk)                ! total precipitation from Hack convection
   real(r8) :: snow_cmf(pcols,begchunk:endchunk)                ! snow from Hack convection

! energy checking variables
   real(r8) :: zero(pcols,begchunk:endchunk)                    ! array of zeros
   real(r8) :: rliq(pcols,begchunk:endchunk)                    ! vertical integral of liquid not yet in q(ixcldliq)
   real(r8) :: rliq2(pcols,begchunk:endchunk)                   ! vertical integral of liquid from shallow scheme
   real(r8) :: cape(pcols,begchunk:endchunk)                   ! cape
   real(r8) :: flx_cnd(pcols,begchunk:endchunk)
   real(r8) :: flx_heat(pcols,begchunk:endchunk)
   logical  :: conserve_energy = .true.       ! flag to carry (QRS,QRL)*dp across time steps

 
   integer jt(pcols,begchunk:endchunk)
   integer maxg(pcols,begchunk:endchunk)
   integer ideep(pcols,begchunk:endchunk)
   integer lengath(begchunk:endchunk)
   real(r8) cldc(pcols,pver,begchunk:endchunk)

   real(r8) nevapr(pcols,pver,begchunk:endchunk)
   real(r8) qme(pcols,pver,begchunk:endchunk)
   real(r8) prain(pcols,pver,begchunk:endchunk)
   real(r8) cflx(pcols,ppcnst,begchunk:endchunk)

#ifdef FILLDEBUG
   real(r8) fakefld(pcols,pver,begchunk:endchunk)
   integer lons(pcols,begchunk:endchunk)
#endif
#ifdef CRM

! Note pritch added new begchunk:endchunk dimension

   real(r8) :: fsntoa(pcols,begchunk:endchunk)        ! Net solar flux at TOA
   real(r8) :: fsntoac(pcols,begchunk:endchunk)       ! Clear sky net solar flux at TOA
   real(r8) :: fsdsc(pcols,begchunk:endchunk)         ! Clear sky flux Shortwave Downwelling Surface
   real(r8) :: fsutoa(pcols,begchunk:endchunk)        ! Flux Shortwave Upwelling TOA
   real(r8) :: fsutoac(pcols,begchunk:endchunk)       ! Clear sky Flux Shortwave Upwelling TOA
   real(r8) :: fsntc(pcols,begchunk:endchunk)         ! Clear sky total column abs solar flux
   real(r8) :: fsnsc(pcols,begchunk:endchunk)         ! Clear sky surface abs solar flux
   real(r8) :: flut(pcols,begchunk:endchunk)          ! Upward flux at top of model
   real(r8) :: flutc(pcols,begchunk:endchunk)         ! Upward Clear Sky flux at top of model
   real(r8) :: flntc(pcols,begchunk:endchunk)         ! Clear sky lw flux at model top
   real(r8) :: flnsc(pcols,begchunk:endchunk)         ! Clear sky lw flux at srf (up-down)
   real(r8) :: lwcf(pcols,begchunk:endchunk)          ! longwave cloud forcing
   real(r8) :: swcf(pcols,begchunk:endchunk)          ! shortwave cloud forcing
   real(r8) :: solin(pcols,begchunk:endchunk)         ! Solar incident flux
   real(r8) :: flwdsc(pcols,begchunk:endchunk)        ! Clear-sky Surface longwave down flux


   type(physics_state) :: state_save(begchunk:endchunk)
   type(physics_tend ) :: tend_save(begchunk:endchunk)
   real(r8) tmp_crm(pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) qc_crm (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) qi_crm (pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) qpc_crm(pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) qpi_crm(pcols, crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) prec_crm(pcols, crm_nx, crm_ny,begchunk:endchunk)
   real(r8) mctot(pcols,pver,begchunk:endchunk)          ! total cloud mass flux
   real(r8) mcup(pcols,pver,begchunk:endchunk)           ! cloud updraft mass flux
   real(r8) mcdn(pcols,pver,begchunk:endchunk)           ! cloud downdraft mass flux
   real(r8) mcuup(pcols,pver,begchunk:endchunk)          ! unsaturated updraft mass flux
   real(r8) mcudn(pcols,pver,begchunk:endchunk)          ! unsaturated downdraft mass flux
   real(r8) crm_qc(pcols,pver,begchunk:endchunk)         ! cloud water
   real(r8) crm_qi(pcols,pver,begchunk:endchunk)         ! cloud ice
   real(r8) crm_qs(pcols,pver,begchunk:endchunk)         ! snow
   real(r8) crm_qg(pcols,pver,begchunk:endchunk)         ! graupel
   real(r8) crm_qr(pcols,pver,begchunk:endchunk)         ! rain
   real(r8) flux_qt(pcols,pver,begchunk:endchunk)        ! nonprecipitating water flux
   real(r8) flux_u(pcols,pver,begchunk:endchunk)        ! x-momentum flux
   real(r8) flux_v(pcols,pver,begchunk:endchunk)        ! y-momentum flux
   real(r8) fluxsgs_qt(pcols,pver,begchunk:endchunk)     ! sgs nonprecipitating water flux
   real(r8) tkez(pcols,pver,begchunk:endchunk)     ! tke profile
   real(r8) tkesgsz(pcols,pver,begchunk:endchunk)     ! sgs tke profile
   real(r8) flux_qp(pcols,pver,begchunk:endchunk)        ! precipitating water flux
   real(r8) precflux(pcols,pver,begchunk:endchunk)       ! precipitation flux
   real(r8) qt_ls(pcols,pver,begchunk:endchunk)        ! water tendency due to large-scale
   real(r8) qt_trans(pcols,pver,begchunk:endchunk)     ! nonprecip water tendency due to transport
   real(r8) qp_trans(pcols,pver,begchunk:endchunk)     ! precip water tendency due to transport
   real(r8) qp_fall(pcols,pver,begchunk:endchunk)      ! precip water tendency due to fall-out
   real(r8) qp_evp(pcols,pver,begchunk:endchunk)       ! precip water tendency due to evap
   real(r8) qp_src(pcols,pver,begchunk:endchunk)       ! precip water tendency due to conversion
   real(r8) t_ls(pcols,pver,begchunk:endchunk)        ! tendency of crm's liwse due to large-scale
   real (r8) u_ls (pcols,pver,begchunk:endchunk)
   real (r8) v_ls (pcols,pver,begchunk:endchunk)
   real(r8) t_rad (crm_nx, crm_ny, crm_nz,begchunk:endchunk) ! rad temperuture
   real(r8) qv_rad(crm_nx, crm_ny, crm_nz,begchunk:endchunk) ! rad vapor
   real(r8) qc_rad(crm_nx, crm_ny, crm_nz,begchunk:endchunk) ! rad cloud water
   real(r8) qi_rad(crm_nx, crm_ny, crm_nz,begchunk:endchunk) ! rad cloud ice
   real(r8) trad(pcols,pver,begchunk:endchunk)
   real(r8) qvrad(pcols,pver,ppcnst,begchunk:endchunk)
   real(r8) fice(pcols,pver,begchunk:endchunk)                  ! Ice fraction from ice and liquid mixing ratios
   real(r8) cldn(pcols,pver,begchunk:endchunk) ! cloud top pdf
   real(r8) cldr(pcols,pver,begchunk:endchunk) ! cloud fraction based on -30dBZ radar reflectivity
   real(r8) cldtop(pcols,pver,begchunk:endchunk)
   real(r8) :: cwp   (pcols,pver,begchunk:endchunk)      ! in-cloud cloud (total) water path
   real(r8) :: gicewp(pcols,pver,begchunk:endchunk)      ! grid-box cloud ice water path
   real(r8) :: gliqwp(pcols,pver,begchunk:endchunk)      ! grid-box cloud liquid water path
   real(r8) :: gwp   (pcols,pver,begchunk:endchunk)      ! grid-box cloud (total) water path
   real(r8) :: tgicewp(pcols,begchunk:endchunk)          ! Vertically integrated ice water path
   real(r8) :: tgliqwp(pcols,begchunk:endchunk)          ! Vertically integrated liquid water path
   real(r8) :: tgwp   (pcols,begchunk:endchunk)          ! Vertically integrated (total) cloud water path
   real(r8) stat_buffer(pcols,19,begchunk:endchunk) ! One-column CRM statistics for the ARM diagnostics


   real(r8) cld_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) cliqwp_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) cicewp_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) rel_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) rei_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) emis_crm(crm_nx, crm_ny, crm_nz,begchunk:endchunk)
   real(r8) fq_isccp_s1(pcols,ntau*npres,begchunk:endchunk) !  the fraction of the model grid box covered by
                                        !  each of the 49 ISCCP D level cloud types
   real(r8) totalcldarea(pcols,begchunk:endchunk) !  the fraction of model grid box columns
                                        !  with cloud somewhere in them.  This should
					!  equal the sum over all entries of fq_isccp
   real(r8) lowcldarea(pcols,begchunk:endchunk), midcldarea(pcols,begchunk:endchunk), hghcldarea(pcols,begchunk:endchunk)
   real(r8) meantaucld(pcols,begchunk:endchunk) !  mean optical thickness (dimensionless)
                                        !  linear averaging in albedo performed.
   real(r8) meanttop(pcols,begchunk:endchunk) !  mean cloud top temp (k) - linear averaging
   real(r8) meanptop(pcols,begchunk:endchunk) !  mean cloud top pressure (mb) - linear averaging
                                        !  in cloud top pressure.
   real(r8) cloudy(pcols,begchunk:endchunk)
   integer  nbreak(begchunk:endchunk)
   logical  doinitial(begchunk:endchunk)
   real(r8) qtot(begchunk:endchunk)
   real(r8) coef(begchunk:endchunk)
   real(r8) kabs(begchunk:endchunk)                   ! longwave absorption coeff (m**2/g)
! CRM column radiation stuff:
   real(r8) qrs_tmp(pcols,pver,begchunk:endchunk)            ! Shortwave heating rate
   real(r8) qrl_tmp(pcols,pver,begchunk:endchunk)            ! Longwave  heating rate
   real(r8) flwdsc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)  ! Clear-sky surface longwave down flux
   real(r8) sols_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)    ! Direct beam solar rad. onto srf (s w)
   real(r8) soll_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)    ! Direct beam solar rad. onto srf (l w)
   real(r8) solsd_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Diffuse solar radiation onto srf ( sw)
   real(r8) solld_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Diffuse solar radiation onto srf (
   real(r8) fsnt_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)    ! Net downward solar flux at model top
   real(r8) fsntc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Clear sky net downward solar flux at model top
   real(r8) fsnsc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Clear sky net downward Shortwave flux Surface
   real(r8) fsutoac_crm(pcols,crm_nx,crm_ny,begchunk:endchunk) ! Clear sky Shortwave Flux Upwelling TOA
   real(r8) flnt_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)    ! Net Upward lw flux at top of model
   real(r8) flntc_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Net Upward Clear Sky flux at top of model
   real(r8) solin_crm(pcols,crm_nx,crm_ny,begchunk:endchunk)   ! Solar incident flux
   real(r8) prectend(pcols,begchunk:endchunk) ! tendency in precipitating water and ice
   real(r8) precstend(pcols,begchunk:endchunk) ! tendency in precipitating ice
   real(r8) wtricesink(pcols,begchunk:endchunk) ! sink of water vapor + cloud water + cloud ice
   real(r8) icesink(pcols,begchunk:endchunk) ! sink of
   real(r8) tau00(begchunk:endchunk)  ! surface stress
   real(r8) wnd(begchunk:endchunk)  ! surface wnd
   real(r8) bflx(pcols,begchunk:endchunk)   ! surface buoyancy flux (Km/s)
   real(r8) taux_crm(pcols,begchunk:endchunk)  ! zonal CRM surface stress perturbation
   real(r8) tauy_crm(pcols,begchunk:endchunk)  ! merid CRM surface stress perturbation
   real(r8) z0m(pcols,begchunk:endchunk)  ! surface momentum roughness length

   real timing_factor(pcols,begchunk:endchunk) ! factor for crm cpu-usage: 1 means no subcycling

   integer ii, jj, mm
   integer iii,lll
#endif
! analysis nudging stuff:

   real(r8) :: aux_s_a   (pcols,pver,begchunk:endchunk)            ! dry static energy of analyses
   real(r8) :: s_tmp (pcols,pver,begchunk:endchunk)            ! dry static energy of analyses
   real(r8) ps_local(pcols,begchunk:endchunk)
   real(r8) dpdt(pcols,begchunk:endchunk)
   logical  lpsntenl

  ! pritch, variables for subregional moisture nudging:

   integer :: nstepsperday
!   real(r8), pointer, dimension(:,:,:) :: qlastdaybuffer
!   real(r8), pointer, dimension (:,:) :: qdaymean_reference
!   real(r8) :: qlastdaymean(pcols,pver)
   real(r8) :: fraction_to_add_over_eight_days,addlimit
   real(r8) :: distance_from_ellipsoid_center, qnudging_timescale_seconds
   real(r8) :: est    (pcols,pver)    ! Saturation vapor pressure
   real(r8) :: qsat   (pcols,pver)    ! saturation specific humidity
#ifdef QRLDAMP
   real (r8) :: newqrl (pcols,pver), dqrl(pcols,pver)
#endif
#ifdef SPFLUXBYPASS 
   real (r8) :: tmp1
#endif
! ---- PRITCH IMPOSED INTERNAL THREAD STAGE 1 -----

   do c=begchunk,endchunk ! Initialize previously acknowledged tphysbc (chunk-level) variable names:
   
     ! MAP ALL-->THIS CHUNK (input args)
     ! (pritch this hacks the variables from now all-chunk received form to their
     ! assumed chunk-specific varnames from original threading paradigm of tphysbc)

     ! These variables used to be passed as individuals from the calling routine.
     ! Now just the whole container packets in_srfflx_state2d and in_surface_state2d are passed.     
     
     ts(:,c) 	    	= in_srfflx_state2d(c)%ts
     sst(:,c) 	    	= in_srfflx_state2d(c)%sst

     precl(:,c) 		= in_surface_state2d(c)%precl
     precc(:,c) 		= in_surface_state2d(c)%precc
     precsl(:,c)    	= in_surface_state2d(c)%precsl
     precsc(:,c)    	= in_surface_state2d(c)%precsc     
     flwds(:,c) 		= in_surface_state2d(c)%flwds(1:pcols)
     lwup(:,c) 	    	= in_srfflx_state2d(c)%lwup(1:pcols)     
     srfrad(:,c)    	= in_surface_state2d(c)%srfrad(1:pcols)
     sols(:,c) 	 	    = in_surface_state2d(c)%sols(1:pcols)
     soll(:,c) 	 	    = in_surface_state2d(c)%soll(1:pcols)
     solsd(:,c)	 		= in_surface_state2d(c)%solsd(1:pcols)
     solld(:,c) 		= in_surface_state2d(c)%solld(1:pcols)          
     asdir(:,c) 		= in_srfflx_state2d(c)%asdir
     asdif(:,c) 		= in_srfflx_state2d(c)%asdif
     aldir(:,c) 		= in_srfflx_state2d(c)%aldir
     aldif(:,c) 		= in_srfflx_state2d(c)%aldif     
     taux(1:pcols,c) 	= in_srfflx_state2d(c)%wsx
     tauy(1:pcols,c) 	= in_srfflx_state2d(c)%wsy  
     shf(1:pcols,c) 	= in_srfflx_state2d(c)%shf  ! surface sensible heat flux (W/m2)
     lhf(1:pcols,c) 	= in_srfflx_state2d(c)%lhf ! surface latent heat flux (W/m2) 

     
     
   ! this next code was originally in tphysbc.F90:... (MS)     
   ! Just updating to generality of column,chunk arrays. 
   
   zero = 0.
!
   lchnk = state(c)%lchnk
   ncol  = state(c)%ncol

   rtdt = 1./ztodt

   nstep = get_nstep()
   calday = get_curr_calday()
!
! Output NSTEP for debugging
!
   timestep(:ncol,c) = nstep
   call outfld ('NSTEP   ',timestep(:,c), pcols, lchnk)

#ifdef FILLDEBUG
   call get_lon_all_p(lchnk, ncol(c), lons)
   do k=1,pver
      do i=1,ncol(c)
         if (nstep > lons(i)) then
            fakefld(i,k,c) = 1.d0/nstep
         else
            fakefld(i,k,c) = fillvalue
         end if
      end do
   end do
   call outfld('FAKEFLD ',fakefld, pcols,lchnk)
#endif

! Associate pointers with physics buffer fields
   itim = pbuf_old_tim_idx()
   ifld = pbuf_get_fld_idx('QCWAT')
   qcwat => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   ifld = pbuf_get_fld_idx('TCWAT')
   tcwat => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   ifld = pbuf_get_fld_idx('LCWAT')
   lcwat => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   ifld = pbuf_get_fld_idx('CLD')
   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

   ifld = pbuf_get_fld_idx('TEOUT')
   teout => pbuf(ifld)%fld_ptr(1,1:pcols,1,lchnk,itim)
   ifld = pbuf_get_fld_idx('QINI')
   qini  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)
   ifld = pbuf_get_fld_idx('TINI')
   tini  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)

!
! Set physics tendencies to 0
   tend(c)%dTdt(:ncol,:pver)  = 0.
   tend(c)%dudt(:ncol,:pver)  = 0.
   tend(c)%dvdt(:ncol,:pver)  = 0.

   call physics_ptend_init (ptend(c)) ! Initialize parameterization tendency structure
!
! Make sure that input tracers are all positive (probably unnecessary)
!
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              ppcnst,qmin  ,state(c)%q )
!
! Setup q and t accumulation fields
!
   dqcond(:ncol,:,:,c) = state(c)%q(:ncol,:,:)
   dtcond(:ncol,:,c)   = state(c)%s(:ncol,:)


   fracis (:ncol,:,1:ppcnst,c) = 1.

!===================================================
! Global mean total energy fixer
!===================================================
   !*** BAB's FV heating kludge *** save the initial temperature
   tini(:ncol,:pver) = state(c)%t(:ncol,:pver)
   if (dycore_is('LR')) then
      call check_energy_fix(state(c), ptend(c), nstep, flx_heat(:,c))
      call physics_update(state(c), tend(c), ptend(c), ztodt)
      call check_energy_chng(state(c), tend(c), "chkengyfix", nstep, ztodt, zero, zero, zero, flx_heat(:,c))
   end if
   qini(:ncol,:pver) = state(c)%q(:ncol,:pver,1)

   call outfld('TEOUT', teout       , pcols, lchnk   )
   call outfld('TEINP', state(c)%te_ini, pcols, lchnk   )
   call outfld('TEFIX', state(c)%te_cur, pcols, lchnk   )

!
!===================================================
! Dry adjustment
!===================================================

! Copy state info for input to dadadj
! This is a kludge, so that dadadj does not have to be correctly reformulated in dry static energy

   ptend(c)%s(:ncol,:pver)   = state(c)%t(:ncol,:pver)
   ptend(c)%q(:ncol,:pver,1) = state(c)%q(:ncol,:pver,1)

   call t_startf ('dadadj')

   call dadadj (lchnk, ncol, state(c)%pmid,  state(c)%pint,  state(c)%pdel,  &
                ptend(c)%s, ptend(c)%q(1,1,1))
   ptend(c)%name  = 'dadadj'
   ptend(c)%ls    = .TRUE.
   ptend(c)%lq(1) = .TRUE.
   ptend(c)%s(:ncol,:)   = (ptend(c)%s(:ncol,:)   - state(c)%t(:ncol,:)  )/ztodt * cpair
   ptend(c)%q(:ncol,:,1) = (ptend(c)%q(:ncol,:,1) - state(c)%q(:ncol,:,1))/ztodt
   call t_stopf ('dadadj')
   call physics_update (state(c), tend(c), ptend(c), ztodt)

#ifdef CRM
! Save the state and tend variables to overwrite conventional physics effects
! leter before calling the superparameterization. Conventional moist
! physics is allowed to compute tendencies due to conventional
! moist physics for diagnostics purposes. -Marat

    state_save(c) = state(c)
    tend_save(c) = tend(c)

#endif

!
!===================================================
! Moist convection
!===================================================
!
! Since the PBL doesn't pass constituent perturbations, they
! are zeroed here for input to the moist convection routine
!
   qpert(:ncol,2:ppcnst,c) = 0.0
!
! Begin with Zhang-McFarlane (1996) convection parameterization
!
   call t_startf ('zm_convr')

   call zm_convr( lchnk,    ncol, &
                  state(c)%t,   state(c)%q,    prec_zmc(:,c),   cnt(:,c),     cnb(:,c),      &
                  pblht(:,c),   state(c)%zm, state(c)%phis,    state(c)%zi,   ptend(c)%q(:,:,1),     &
                  ptend(c)%s, state(c)%pmid,   state(c)%pint,  state(c)%pdel,       &
                   .5*ztodt,cmfmc(:,:,c),    cmfcme(:,:,c),             &
                  tpert(:,c),   dlf(:,:,c),      pflx(:,:,c),    zdu(:,:,c),     zmrprd(:,:,c),   &
                  mu2(:,:,c),      md2(:,:,c),     du2(:,:,c),     eu2(:,:,c),     ed2(:,:,c),      &
                  dp(:,:,c),       dsubcld(:,c), jt(:,c),      maxg(:,c),    ideep(:,c),    &
                  lengath(c), icwmr1(:,:,c),   rliq(:,c), cape(:,c)    )
                  
  
!
! Convert mass flux from reported mb/s to kg/m^2/s
!
   cmfmc(:ncol,:pver,c) = cmfmc(:ncol,:pver,c) * 100./gravit

!  Momentum transport by deep convection, Marat Khairoutdinov, May 2005
#ifdef MOMTRANS
   ptend(c)%u(:,:) = 0.
   ptend(c)%v(:,:) = 0.
   do i=1,ncol 
     if(prec_zmc(i)*86400.*1000..gt.20.) then   
!       print*, 'prec_zmc=',prec_zmc(i)*86400.*1000., int(cnt(i)), state(c)%zm(i,int(cnt(i)))
       call convmom(state(c)%zm(i,:),state(c)%zi(i,:),state(c)%pdel(i,:),state(c)%u(i,:),cape(:,c), cmfmc(i,:,c),&
                    state(c)%zm(i,int(cnt(i))), pver, ptend(c)%u(i,:))  
       call convmom(state(c)%zm(i,:),state(c)%zi(i,:),state(c)%pdel(i,:),state(c)%v(i,:),cape(:,c), cmfmc(i,:,c),&
                    state(c)%zm(i,int(cnt(i))), pver, ptend(c)%v(i,:))  
     end if
   end do
   ptend(c)%lu    = .TRUE.
   ptend(c)%lv    = .TRUE.
   call outfld('UCONVMOM',ptend(c)%u,pcols   ,lchnk   )
   call outfld('VCONVMOM',ptend(c)%v,pcols   ,lchnk   )
#endif

   ptend(c)%name  = 'zm_convr'
   ptend(c)%ls    = .TRUE.
   ptend(c)%lq(1) = .TRUE.
   cmfsl (:ncol,:,c) = 0. ! This is not returned from zm, hence it is zeroed.
   cmflq (:ncol,:,c) = 0. ! This is not returned from zm, hence it is zeroed.

   ftem(:ncol,:pver,c) = ptend(c)%s(:ncol,:pver)/cpair
   call outfld('ZMDT    ',ftem(:,:,c),pcols   ,lchnk   )
   call outfld('ZMDQ    ',ptend(c)%q(1,1,1) ,pcols   ,lchnk   )
   call t_stopf('zm_convr')

   call physics_update(state(c), tend(c), ptend(c), ztodt)
!
! Determine the phase of the precipitation produced and add latent heat of fusion
! Evaporate some of the precip directly into the environment (Sundqvist)

   call zm_conv_evap(state(c), ptend(c), zmrprd(:,:,c), cld, ztodt, prec_zmc(:,c), snow_zmc(:,c), .false.)
   call physics_update(state(c), tend(c), ptend(c), ztodt)
! Check energy integrals, including "reserved liquid"
   flx_cnd(:ncol,c) = prec_zmc(:ncol,c) + rliq(:ncol,c)
   call check_energy_chng(state(c), tend(c), "zm_evap", nstep, ztodt, zero, flx_cnd(:,c), snow_zmc(:,c), zero)

! Transport cloud water and ice only
!
   call cnst_get_ind('CLDLIQ', ixcldliq)
   call cnst_get_ind('CLDICE', ixcldice)
   ptend(c)%name = 'convtran1'
   ptend(c)%lq(ixcldice) = .true.
   ptend(c)%lq(ixcldliq) = .true.
   call t_startf ('convtran1')

   call convtran (lchnk,                                        &
                  ptend(c)%lq,state(c)%q, ppcnst,  mu2(:,:,c),     md2(:,:,c),   &
                  du2(:,:,c),     eu2(:,:,c),     ed2(:,:,c),     dp(:,:,c),      dsubcld(:,c),  &
                  jt(:,c),      maxg(:,c),    ideep(:,c),   1,       lengath(c),  &
                  nstep,   fracis(:,:,:,c),  ptend(c)%q   )
   call t_stopf ('convtran1')
   call physics_update (state(c), tend(c), ptend(c), ztodt)

!
! Call Hack (1994) convection scheme to deal with shallow/mid-level convection
!
   call t_startf('cmfmca')
   tpert2(:ncol,c) =0.
   qpert2(:ncol,:,c) = qpert(:ncol,:,c)  ! BAB Why is this not zero, if tpert2=0???
 call cmfmca (lchnk,   ncol, &
                nstep,   ztodt,   state(c)%pmid,  state(c)%pdel,   &
                state(c)%rpdel,   state(c)%zm,      tpert2(:,c),  qpert2(:,:,c),  state(c)%phis,     &
                pblht(:,c),   state(c)%t,   state(c)%q,   ptend(c)%s,   ptend(c)%q,      &
                cmfmc2(:,:,c),  cmfdqr2(:,:,c), cmfsl2(:,:,c),  cmflq2(:,:,c),  prec_cmf(:,c),   &
                dlf2(:,:,c),     cnt2(:,c),    cnb2(:,c),    icwmr2(:,:,c)   , rliq2(:,c))
   ptend(c)%name  = 'cmfmca'
   ptend(c)%ls    = .TRUE.
   ptend(c)%lq(:) = .TRUE.


! Add shallow cloud water detrainment to cloud water detrained from ZM
   dlf(:ncol,:pver,c) = dlf(:ncol,:pver,c) + dlf2(:ncol,:pver,c)
   rliq(:ncol,c) = rliq(:ncol,c) + rliq2(:ncol,c)
   
   ftem(:ncol,:pver,c) = ptend(c)%s(:ncol,:pver)/cpair
   call outfld('CMFDT   ',ftem(:,:,c)    ,pcols   ,lchnk   )
   call outfld('CMFDQ   ',ptend(c)%q(1,1,1),pcols   ,lchnk   )
   call t_stopf('cmfmca')
   call physics_update (state(c), tend(c), ptend(c), ztodt)
   
!
! Determine the phase of the precipitation produced and add latent heat of fusion
   call zm_conv_evap(state(c), ptend(c), cmfdqr2(:,:,c), cld, ztodt, prec_cmf(:,c), snow_cmf(:,c), .true.)
   call physics_update(state(c), tend(c), ptend(c), ztodt)
   flx_cnd(:ncol,c) = prec_cmf(:ncol,c) + rliq2(:ncol,c)
   call check_energy_chng(state(c), tend(c), "hk_evap", nstep, ztodt, zero, flx_cnd(:,c), snow_cmf(:,c), zero)
!
! Merge shallow/mid-level output with prior results from Zhang-McFarlane
!
   do i=1,ncol
      if (cnt2(i,c) < cnt(i,c)) cnt(i,c) = cnt2(i,c)
      if (cnb2(i,c) > cnb(i,c)) cnb(i,c) = cnb2(i,c)
   end do
!
   cmfmc (:ncol,:pver,c) = cmfmc (:ncol,:pver,c) + cmfmc2 (:ncol,:pver,c)
   cmfsl (:ncol,:pver,c) = cmfsl (:ncol,:pver,c) + cmfsl2 (:ncol,:pver,c)
   cmflq (:ncol,:pver,c) = cmflq (:ncol,:pver,c) + cmflq2 (:ncol,:pver,c)
   call outfld('CMFMC' , cmfmc(:,:,c)  , pcols, lchnk)
!  output new partition of cloud condensate variables, as well as precipitation 
   call outfld('QC      ',dlf2 (:,:,c)          ,pcols   ,lchnk   )
   call outfld('PRECSH  ',prec_cmf(:,c)      ,pcols   ,lchnk       )
   call outfld('CMFDQR', cmfdqr2(:,:,c), pcols, lchnk)
   call outfld('CMFSL' , cmfsl(:,:,c)  , pcols, lchnk)
   call outfld('CMFLQ' , cmflq(:,:,c)  , pcols, lchnk)
   call outfld('DQP'   , dlf2(:,:,c)    , pcols, lchnk)

! Allow the cloud liquid drops and ice particles to sediment
! Occurs before adding convectively detrained cloud water, because the phase of the
! of the detrained water is unknown.


   call t_startf('cldwat_sediment')
   call cldcond_sediment(state(c), ptend(c), ztodt,cld, in_icefrac(:,c), in_landfrac(:,c), in_ocnfrac(:,c), prec_sed(:,c), &
                         snow_sed(:,c), in_landm(:,c), snowh(:,c))

   call physics_update(state(c), tend(c), ptend(c), ztodt)
   call t_stopf('cldwat_sediment')

! check energy integrals
   call check_energy_chng(state(c), tend(c), "cldwat_sediment", nstep, ztodt, zero, prec_sed(:,c), snow_sed(:,c), zero)

! Put the detraining cloud water from convection into the cloud and environment. 
   call t_startf('cldwat_detrain')
   call cldcond_zmconv_detrain(dlf(:,:,c), cld, state(c), ptend(c))
   call physics_update(state(c), tend(c), ptend(c), ztodt)
   call t_stopf('cldwat_detrain')

! check energy integrals, reserved liquid has now been used
   flx_cnd(:ncol,c) = -rliq(:ncol,c)
   call check_energy_chng(state(c), tend(c), "cldwat_detrain", nstep, ztodt, zero, flx_cnd(:,c), zero, zero)
!
! cloud fraction after transport and convection,
! derive the relationship between rh and cld from 
! the employed cloud scheme
!
   call t_startf('cldnrh')
!   if(lchnk.eq.478.and.nstep.lt.96) then
!        i = 10
!        write(22,*) &
!        state(c)%t(i,:),   state(c)%q(i,:,:), cnt(i),     cnb(i),cmfmc(i,:),   cmfmc2(i,:),zdu(i,:),dlf(i,:)
!   end if
   call cldnrh(lchnk,   ncol,                                &
               state(c)%pmid,    state(c)%t,   state(c)%q(1,1,1),   state(c)%omega, &
                 cnt(:,c),     cnb(:,c),     cld,    clc(:,c),     state(c)%pdel,   &
               cmfmc(:,:,c),   cmfmc2(:,:,c), in_landfrac(:,c), snowh(:,c) &
,   concld(:,:,c),  cldst(:,:,c), &
               in_srfflx_state2d(c)%ts,      in_srfflx_state2d(c)%sst, state(c)%pint(1,pverp),       zdu(:,:,c),  in_ocnfrac(:,c), &
               rhdfda(:,:,c),   rhu00(:,:,c) , state(c)%phis)
   call t_stopf('cldnrh')
#ifdef CRM
   call outfld('_CONCLD  ',concld(:,:,c), pcols,lchnk)
   call outfld('_CLDST   ',cldst(:,:,c),  pcols,lchnk)
   call outfld('_CNVCLD  ',clc(:,c),    pcols,lchnk)
#else
   call outfld('CONCLD  ',concld(:,:,c), pcols,lchnk)
   call outfld('CLDST   ',cldst(:,:,c),  pcols,lchnk)
   call outfld('CNVCLD  ',clc(:,c),    pcols,lchnk)
#endif
! cloud water and ice parameterizations
   call t_startf('cldwat_tend')
   call cldcond_tend(state(c), ptend(c), ztodt, &
       tcwat, qcwat, lcwat, prec_pcw(:,c), snow_pcw(:,c), in_icefrac(:,c), rhdfda(:,:,c), rhu00(:,:,c), cld, nevapr(:,:,c), prain(:,:,c), qme(:,:,c), snowh(:,c))
   call physics_update (state(c), tend(c), ptend(c), ztodt)
!   if(lchnk.eq.478.and.nstep.le.96) then
!        i = 10
!	write(22,*) state%t(i,:),   state%q(i,:,:), cld(i,:), prec_zmc(i), prec_cmf(i), prec_sed(i), &
!          prec_pcw(i)
!
!	if(nstep.eq.96) close(22)
!   end if
   call t_stopf('cldwat_tend')

! check energy integrals
   call check_energy_chng(state(c), tend(c), "cldwat_tend", nstep, ztodt, zero, prec_pcw(:,c), snow_pcw(:,c), zero)

! Save off q and t after cloud water
   do k=1,pver
      qcwat(:ncol,k) = state(c)%q(:ncol,k,1)
      tcwat(:ncol,k) = state(c)%t(:ncol,k)
      lcwat(:ncol,k) = state(c)%q(:ncol,k,ixcldice) + state(c)%q(:ncol,k,ixcldliq)
   end do
!
!  aerosol wet chemistry determines scavenging fractions, and transformations
   call get_lat_all_p(lchnk, ncol, lat(:,c))
   call get_lon_all_p(lchnk, ncol, lon(:,c))
   call get_rlat_all_p(lchnk, ncol, clat(:,c))
   conicw(:ncol,:,c) = icwmr1(:ncol,:,c) + icwmr2(:ncol,:,c)
   cmfdqrt(:ncol,:,c) = zmrprd(:ncol,:,c) + cmfdqr2(:ncol,:,c)
   call aerosol_wet_intr (state(c), ptend(c), cflx(:,:,c), nstep, ztodt, lat(:,c), clat(:,c), lon,&
        qme(:,:,c), prain(:,:,c), &
       nevapr(:,:,c), cldc(:,:,c), cld, fracis(:,:,:,c), calday, cmfdqrt(:,:,c), conicw(:,:,c))
   call physics_update (state(c), tend(c), ptend(c), ztodt)

!
!     Convective transport of all trace species except water vapor and
!     cloud liquid and ice done here because we need to do the scavenging first
!     to determine the interstitial fraction.
!
   ptend(c)%name  = 'convtran2'
   ptend(c)%lq(:) = .true.
   ptend(c)%lq(ixcldice) = .false.
   ptend(c)%lq(ixcldliq) = .false.
   call t_startf ('convtran2')
   call convtran (lchnk,                                           &
                  ptend(c)%lq,state(c)%q, ppcnst,     mu2,     md2,      &
                  du2(:,:,c),     eu2(:,:,c),     ed2(:,:,c),       dp(:,:,c),      dsubcld(:,c),  &
                  jt(:,c),      maxg(:,c),    ideep(:,c),      1,       lengath(c),  &
                  nstep,   fracis(:,:,:,c),  ptend(c)%q)
   call t_stopf ('convtran2')

   call physics_update (state(c), tend(c), ptend(c), ztodt)
!
! Compute rates of temperature and constituent change due to moist processes
!
   dtcond(:ncol,:,c) = (state(c)%s(:ncol,:) - dtcond(:ncol,:,c))*rtdt / cpair
   dqcond(:ncol,:,:,c) = (state(c)%q(:ncol,:,:) - dqcond(:ncol,:,:,c))*rtdt
   call outfld('DTCOND  ',dtcond(:,:,c), pcols   ,lchnk   )
   do m=1,ppcnst
      call outfld(dcconnam(m),dqcond(1,1,m,c),pcols   ,lchnk )
      call outfld('DQCOND  ',dqcond(1,1,m,c),pcols   ,lchnk   )
   end do

! Compute total convective and stratiform precipitation and snow rates
   do i=1,ncol
      precc (i,c) = prec_zmc(i,c) + prec_cmf(i,c)
      precl (i,c) = prec_sed(i,c) + prec_pcw(i,c)
      precsc(i,c) = snow_zmc(i,c) + snow_cmf(i,c)
      precsl(i,c) = snow_sed(i,c) + snow_pcw(i,c)
! jrm These checks should not be necessary if they exist in the parameterizations
      if(precc(i,c).lt.0.) precc(i,c)=0.
      if(precl(i,c).lt.0.) precl(i,c)=0.
      if(precsc(i,c).lt.0.) precsc(i,c)=0.
      if(precsl(i,c).lt.0.) precsl(i,c)=0.
      if(precsc(i,c).gt.precc(i,c)) precsc(i,c)=precc(i,c)
      if(precsl(i,c).gt.precl(i,c)) precsl(i,c)=precl(i,c)
! end jrm
   end do
   in_prcsnw(:ncol,c) = precsc(:ncol,c) + precsl(:ncol,c)   ! total snowfall rate: needed by slab ocean model
!
!===================================================
! Moist physical parameteriztions complete: 
! send dynamical variables, and derived variables to history file
!===================================================
!
#ifndef CRM
   call diag_dynvar (lchnk, ncol, state(c))
#endif
!
!===================================================
! Radiation computations
!===================================================
!
! Cosine solar zenith angle for current time step
!
   call get_rlat_all_p(lchnk, ncol, clat(:,c))
   call get_rlon_all_p(lchnk, ncol, clon(:,c))
   call zenith (calday, clat(:,c), clon(:,c), coszrs(:,c), ncol)

   if (dosw .or. dolw) then

! Compute cloud water/ice paths and optical properties for input to radiation
      call t_startf('cldoptics')
      call param_cldoptics_calc(1, ncol, state(c), cld, in_landfrac(:,c), in_landm(:,c),in_icefrac(:,c), &
  cicewp(:,:,c), cliqwp(:,:,c), emis(:,:,c), rel(:,:,c), rei(:,:,c), pmxrgn(:,:,c), nmxrgn(:,c), snowh(:,c))
      call t_stopf('cldoptics')
!
! Complete radiation calculations
      call t_startf ('radctl')
               call radctl (lchnk, 1, ncol, lwup(:,c), emis(:,:,c), state(c)%pmid,             &
                   state(c)%pint, state(c)%lnpmid, state(c)%lnpint, state(c)%t, state(c)%q,   &
                   cld, cicewp(:,:,c), cliqwp(:,:,c), coszrs(:,c), asdir(:,c), asdif(:,c),               &
                   aldir(:,c), aldif(:,c), pmxrgn(:,:,c), nmxrgn(:,c), in_fsns(:,c), in_fsnt(:,c),in_flns(:,c),in_flnt(:,c), &
                   qrs(:,:,c), qrl(:,:,c), flwds(:,c), rel(:,:,c), rei(:,:,c), &
                   sols(:,c), soll(:,c), solsd(:,c), solld(:,c),                  &
                   in_landfrac(:,c), state(c)%zm, state(c), fsds(:,c) &
#ifdef CRM
                  ,fsntoa(:,c) ,fsntoac(:,c) ,fsdsc(:,c)   ,flwdsc(:,c)  ,fsntc(:,c) &
                  ,fsnsc(:,c)   , &
                  fsutoa(:,c) ,fsutoac(:,c) ,flut(:,c)    ,flutc(:,c)   ,flntc(:,c)   ,flnsc(:,c)   ,solin(:,c)   , &
                   .true., doabsems, dosw, dolw &
#endif
             )         
      call t_stopf ('radctl')


!
! Cloud cover diagnostics
! radctl can change pmxrgn and nmxrgn so cldsav needs to follow
! radctl.
!
      call cldsav (lchnk, 1, ncol, cld, state(c)%pmid, cltot(:,c), &
                   cllow(:,c), clmed(:,c), clhgh(:,c), nmxrgn(:,c), pmxrgn(:,:,c))
!
! Dump cloud field information to history tape buffer (diagnostics)
!
#ifdef CRM
      call outfld('_CLOUD  ',cld,  pcols,lchnk)
      call outfld('_CLDTOT ',cltot(:,c)  ,pcols,lchnk)
      call outfld('_CLDLOW ',cllow(:,c)  ,pcols,lchnk)
      call outfld('_CLDMED ',clmed(:,c)  ,pcols,lchnk)
      call outfld('_CLDHGH ',clhgh(:,c)  ,pcols,lchnk)
#else
      call outfld('CLDTOT  ',cltot(:,c)  ,pcols,lchnk)
      call outfld('CLDLOW  ',cllow(:,c)  ,pcols,lchnk)
      call outfld('CLDMED  ',clmed(:,c)  ,pcols,lchnk)
      call outfld('CLDHGH  ',clhgh(:,c)  ,pcols,lchnk)
      call outfld('CLOUD   ',cld    ,pcols,lchnk)
#endif
      if (doisccp) then
         call ccm_isccp(lchnk, ncol, state(c)%pmid, state(c)%pint, state(c)%q, &
                        state(c)%t, ts(:,c), concld(:,:,c),   &
                        cld, cliqwp(:,:,c), cicewp(:,:,c), rel(:,:,c), rei(:,:,c), emis(:,:,c), cltot(:,c), coszrs(:,c))
      end if
   else

! convert radiative heating rates from Q*dp to Q for energy conservation
      if (conserve_energy) then
         do k =1 , pver
            do i = 1, ncol
               qrs(i,k,c) = qrs(i,k,c)/state(c)%pdel(i,k)
               qrl(i,k,c) = qrl(i,k,c)/state(c)%pdel(i,k)
            end do
         end do
      end if

   end if

  end do    ! ------ PRITCH END INTERNALLY THREADED CHUNK LOOP STAGE 1 -----


#ifdef CRM
!========================================================
!========================================================
!=   Warning! You're entering a no-spin zone ! ==========
!========================================================
!========================================================
!========================================================
!========================================================
!  CRM (Superparameterization).
! Author: Marat Khairoutdinov marat@atmos.colostate.edu
!========================================================


! Forget all changes to the state due to conventional physics above:


! Initialize stuff:

call t_startf ('crm')
do c=begchunk,endchunk
  state(c) = state_save(c)
  tend(c) = tend_save(c)
end do

   if(is_first_step() .and. .not. crminitread ) then
	  do c=begchunk,endchunk ! ---- PRITCH BEGIN CRM-CENTRIC CHUNK LOOP -----
	   lchnk = state(c)%lchnk
	   ncol  = state(c)%ncol 
	   ifld = pbuf_get_fld_idx('CLD')
	   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
	   spcld(:,:,c) = cld(:,:)	   
!      call check_energy_timestep_init(state, tend, pbuf)

      do k=1,crm_nz
        do i=1,ncol
             m = pver-k+1
#ifndef CRM3D
#ifdef SP_DIR_NS
             u_crm  (i,:,:,k,c) = state(c)%v(i,m)
             v_crm  (i,:,:,k,c) = state(c)%u(i,m)
#else
             u_crm  (i,:,:,k,c) = state(c)%u(i,m)  
             v_crm  (i,:,:,k,c) = state(c)%v(i,m)  
#endif
#endif
#ifdef CRM3D
               u_crm  (i,:,:,k,c) = state(c)%u(i,m)
               v_crm  (i,:,:,k,c) = state(c)%v(i,m)
#endif
             w_crm  (i,:,:,k,c) = 0.
             t_crm  (i,:,:,k,c) = state(c)%t(i,m)
             q_crm  (i,:,:,k,c) = state(c)%q(i,m,1)+state(c)%q(i,m,ixcldliq)+state(c)%q(i,m,ixcldice)
             qn_crm  (i,:,:,k,c) = state(c)%q(i,m,ixcldliq)+state(c)%q(i,m,ixcldice)
             qp_crm (i,:,:,k,c) = 0.
             qrs_crm(i,:,:,k,c) = (qrs(i,m,c)) /cpair
             qrl_crm(i,:,:,k,c) = (qrl(i,m,c)) /cpair
             qc_crm (i,:,:,k,c) = 0.
             qi_crm (i,:,:,k,c) = 0.
             qpc_crm (i,:,:,k,c) = 0.
             qpi_crm (i,:,:,k,c) = 0.
          end do
      end do
      in_rad_buffer(:,:,c) = 0.
! use radiation from grid-cell mean radctl on first time step
      coef(c) = dble(crm_nx*crm_ny)
      do i=1,ncol
         prec_crm (i,:,:,c) = 0.
         do k=1,pver
           qrs1(i,k,c) = qrs(i,k,c) * coef(c) * state_save(c)%pdel(i,k)
           qrl1(i,k,c) = qrl(i,k,c) * coef(c) * state_save(c)%pdel(i,k)
         end do  
         in_rad_buffer(i,1,c) =  solin(i,c) * coef(c)
         in_rad_buffer(i,2,c) =  fsds(i,c) * coef(c)
         in_rad_buffer(i,3,c) =  fsdsc(i,c) * coef(c)
         in_rad_buffer(i,4,c) =  in_fsnt(i,c) * coef(c)
         in_rad_buffer(i,5,c) =  in_fsns(i,c) * coef(c)
         in_rad_buffer(i,6,c) =  fsntc(i,c) * coef(c)
         in_rad_buffer(i,7,c) =  fsnsc(i,c) * coef(c)
         in_rad_buffer(i,8,c) =  fsntoa(i,c) * coef(c)
         in_rad_buffer(i,9,c) =  fsntoac(i,c) * coef(c)
         in_rad_buffer(i,10,c) = sols(i,c) * coef(c)
         in_rad_buffer(i,11,c) = soll(i,c) * coef(c)
         in_rad_buffer(i,12,c) = solsd(i,c) * coef(c)
         in_rad_buffer(i,13,c) = solld(i,c) * coef(c)
         in_rad_buffer(i,14,c) = in_flnt(i,c) * coef(c)
         in_rad_buffer(i,15,c) = flut(i,c) * coef(c)
         in_rad_buffer(i,16,c) = flntc(i,c) * coef(c)
         in_rad_buffer(i,17,c) = flutc(i,c) * coef(c)
         in_rad_buffer(i,18,c) = in_flns(i,c) * coef(c)
         in_rad_buffer(i,19,c) = flnsc(i,c) * coef(c)
         in_rad_buffer(i,20,c) = flwds(i,c) * coef(c)
         in_rad_buffer(i,21,c) = fsutoa(i,c) * coef(c)
         in_rad_buffer(i,22,c) = fsutoac(i,c) * coef(c)
         in_rad_buffer(i,23,c) = flwdsc(i,c) * coef(c)
         fsds_crm(i,:,:,c) = fsds(i,c)
         fsns_crm(i,:,:,c) = in_fsns(i,c)
         fsdsc_crm(i,:,:,c) = fsdsc(i,c)
         fsntoa_crm(i,:,:,c) = fsntoa(i,c)
         fsntoac_crm(i,:,:,c) = fsntoac(i,c)
         fsutoa_crm(i,:,:,c) = fsutoa(i,c)
         flwds_crm(i,:,:,c) = flwds(i,c)
         flns_crm(i,:,:,c) = in_flns(i,c)
         flnsc_crm(i,:,:,c) = flnsc(i,c)
         flnt_crm(i,:,:,c) = in_flnt(i,c)
         flntc_crm(i,:,:,c) = flntc(i,c)
         flut_crm(i,:,:,c) = flut(i,c)
         flutc_crm(i,:,:,c) = flutc(i,c)
         lwcf(i,c) = flutc(i,c) - flut(i,c)
         swcf(i,c) = fsntoa(i,c) - fsntoac(i,c)
      enddo
      lwcf(:,c) = flutc(:,c) - flut(:,c)
      swcf(:,c) = fsntoa(:,c) - fsntoac(:,c)
      ptend(c)%q(:,:,1) = 0.
      ptend(c)%q(:,:,ixcldliq) = 0.
      ptend(c)%q(:,:,ixcldice) = 0.
      ptend(c)%s(:,:) = 0.
      precc(:,c) = 0.
      precl(:,c) = 0.
      precsc(:,c) = 0.
      precsl(:,c) = 0.
      cltot(:,c) = 0.
      clhgh(:,c) = 0.
      clmed(:,c) = 0.
      cllow(:,c) = 0.
!      cld(:,:) = 0.
      spcld(:,:,c) = 0.
      cldr(:,:,c) = 0.
      cldtop(:,:,c) = 0.
      gicewp(:,:,c)=0
      gliqwp(:,:,c)=0
      stat_buffer(:,:,c) = 0.
      mctot(:,:,c) = 0.
      mcup(:,:,c) = 0.
      mcdn(:,:,c) = 0.
      mcuup(:,:,c) = 0.
      mcudn(:,:,c) = 0.
      crm_qc(:,:,c) = 0.
      crm_qi(:,:,c) = 0.
      crm_qs(:,:,c) = 0.
      crm_qg(:,:,c) = 0.
      crm_qr(:,:,c) = 0.
      flux_qt(:,:,c) = 0.
      flux_u(:,:,c) = 0.
      flux_v(:,:,c) = 0.
      fluxsgs_qt(:,:,c) = 0.
      tkez(:,:,c) = 0.
      tkesgsz(:,:,c) = 0.
      flux_qp(:,:,c) = 0.
      precflux(:,:,c) = 0.
      qt_ls(:,:,c) = 0.
      qt_trans(:,:,c) = 0.
      qp_trans(:,:,c) = 0.
      qp_fall(:,:,c) = 0.
      qp_evp(:,:,c) = 0.
      qp_src(:,:,c) = 0.
      z0m(:,c) = 0.
      taux_crm(:,c) = 0.
      tauy_crm(:,c) = 0.
      t_ls(:,:,c) = 0.
      u_ls(:,:,c) = 0.
      v_ls(:,:,c) = 0.
      end do ! pritch begchunk etc.

   else
   
#ifdef SPFLUXBYPASS
 ! pritch -- apply surface flux perturbations to lowest level DSE,
 ! constituents here, right before superparameterization, insated of
 ! instead of in vertical diffusion routine. To avoid exposiing dycore to a
 ! bottom-centric physics tendency profile component, and to allow SP to do
 ! the job of diffusing the flux infusion.
  do c=begchunk,endchunk
    ncol  = state(c)%ncol
    ptend(c)%lu = .false.
    ptend(c)%lv = .false.
    ptend(c)%ls = .true.
    ptend(c)%lq = .false.
    ptend(c)%lq(1) = .true.
    do i=1,ncol
      tmp1 = gravit*state(c)%rpdel(i,pver)
      ptend(c)%s(i,:) = 0.
      ptend(c)%s(i,pver) = tmp1*shf(i,c) 
      ptend(c)%q(i,:,1) = 0.
      ptend(c)%q(i,pver,1) = tmp1*cflx(i,1,c)
    end do
    call physics_update (state(c), tend(c), ptend(c), ztodt)
  end do
#endif
! ================= BEGIN THREADED ZONE OF MOST WORK ===================

! INSERT PRITCH MY MAIN TARGET INTERNALLY TRHEADED LOOP STARTS HERE:
! INSERT add omp directives
! INSERT add MIC offload syntax
! INSERT Make c,k,m,i,lchnk,ncol,coef,ii,jj,mm,ifld,cld private.

!$OMP PARALLEL DO PRIVATE (C,K,M,I,LCHNK,NCOL,II,JJ,MM,IFLD)
   do c=begchunk,endchunk ! pritch new chunk loop
      ncol  = state(c)%ncol
      lchnk = state(c)%lchnk
      ! INSERT warning cld and its pointer status may cause trouble on threading...
      
   !   ifld = pbuf_get_fld_idx('CLD')
   !   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

      ptend(c)%q(:,:,1) = 0.  ! necessary?
      ptend(c)%q(:,:,ixcldliq) = 0.
      ptend(c)%q(:,:,ixcldice) = 0.
      ptend(c)%s(:,:) = 0. ! necessary?
      trad(:ncol,:,c)  = state(c)%t(:ncol,:)
      qvrad(:ncol,:,1,c) = state(c)%q(:ncol,:,1)

     cwp(:,:,c)   = 0.
     fice(:,:,c)  = 0.
     cldn(:,:,c)  = 0.
     emis(:,:,c)  = 0.
     gicewp(:,:,c) = 0.
     gliqwp(:,:,c) = 0.
     cicewp(:,:,c) = 0.
     cliqwp(:,:,c) = 0.
     stat_buffer(:ncol,:,c) = 0.
     cltot(:,c) = 0.
     clhgh(:,c) = 0.
     clmed(:,c) = 0.
     cllow(:,c) = 0.

! Recall previous step's rad statistics for correct averaging on a current time step::

     qrs(:,:,c) = qrs1(:,:,c)
     qrl(:,:,c) = qrl1(:,:,c)
     solin  (:,c)   = in_rad_buffer(:,1,c)
     fsds   (:,c)   = in_rad_buffer(:,2,c)
     fsdsc  (:,c)   = in_rad_buffer(:,3,c)
     in_fsnt   (:,c)   = in_rad_buffer(:,4,c)
     in_fsns   (:,c)   = in_rad_buffer(:,5,c)
     fsntc  (:,c)   = in_rad_buffer(:,6,c)
     fsnsc  (:,c)   = in_rad_buffer(:,7,c)
     fsntoa (:,c)   = in_rad_buffer(:,8,c)
     fsntoac(:,c)   = in_rad_buffer(:,9,c)
     sols   (:,c)   = in_rad_buffer(:,10,c)
     soll   (:,c)   = in_rad_buffer(:,11,c)
     solsd  (:,c)   = in_rad_buffer(:,12,c)
     solld  (:,c)   = in_rad_buffer(:,13,c)
     in_flnt   (:,c)   = in_rad_buffer(:,14,c)
     flut   (:,c)   = in_rad_buffer(:,15,c)
     flntc  (:,c)   = in_rad_buffer(:,16,c)
     flutc  (:,c)   = in_rad_buffer(:,17,c)
     in_flns   (:,c)   = in_rad_buffer(:,18,c)
     flnsc  (:,c)   = in_rad_buffer(:,19,c)
     flwds  (:,c)   = in_rad_buffer(:,20,c)
     fsutoa (:,c)   = in_rad_buffer(:,21,c)
     fsutoac(:,c)   = in_rad_buffer(:,22,c)
     flwdsc (:,c)   = in_rad_buffer(:,23,c)

! Initialize save-buffers

     in_rad_buffer(:,:,c) = 0.
     qrs1(:,:,c) = 0.
     qrl1(:,:,c) = 0.
!
! superparameterization radiation cycling starts here:
!
! pritch added for gentine---- 1/17
     call outfld('TBSP   ',state(c)%t(:,:)     ,pcols   ,lchnk   )
     call outfld('QBSP   ',state(c)%q(:,:,1)     ,pcols   ,lchnk   )
     call outfld('QIBSP  ',state(c)%q(:,:,ixcldice)     ,pcols   ,lchnk   )
     call outfld('QCBSP  ',state(c)%q(:,:,ixcldliq)     ,pcols   ,lchnk   )
     call outfld('UBSP   ',state(c)%u(:,:),pcols   ,lchnk   )
     call outfld('VBSP   ',state(c)%v(:,:),pcols   ,lchnk   )
! ----
     do i = 1,ncol

        doinitial(c) = .true.
        nbreak(c) = nint(ztodt/RADTIME)
#ifdef CRMACCEL
        if (do_accel) then
      nbreak(c) = nbreak(c)/(crm_accel_factor+1)
          ! note that crm_accel_factor = 0 ==> no accel.
          ! crm_accel_factor = 1 ==> 2x accel, 
          ! crm_accel_factor = 2 ==> tend -> tend + 2*tend = 3x.

          ! If 2x acceleration, call radiation 1/2 as many time steps.
          ! preserves frequency of radiative transfer for cost saving.
          if (mod(nbreak(c),1) .ne. 0) then
             write (6,*) 'ENDRUN: CRMACCEL: nbreak non integer =',nbreak
             write (6,*) 'crm_accel_factor was',crm_accel_factor
             stop
          end if
        end if
#endif
        do m=1,crm_nz
           k = pver-m+1
           qrs_crm(i,:,:,m,c) = qrs_crm(i,:,:,m,c) / state(c)%pdel(i,k) ! for energy conservation
           qrl_crm(i,:,:,m,c) = qrl_crm(i,:,:,m,c) / state(c)%pdel(i,k) ! for energy conservation
        end do

		tau00(c) = sqrt(taux(i,c)**2 + tauy(i,c)**2)
		wnd(c) = sqrt(state(c)%u(i,pver)**2 + state(c)%v(i,pver)**2)
		
        bflx(i,c) = shf(i,c)/cpair + 0.61*state(c)%t(i,pver)*lhf(i,c)/latvap
        
        ! Units of bflx? J/s/m2 / (J/kg/K) = K kg / m2 /s  ; weird that inside crm.F it says (K m/s)?
        do  mm=1,nbreak(c)
	  
   call t_startf ('crm_call')
        
	  call crm (c, i, &
               state(c)%t(i,:), state(c)%q(i,:,1), state(c)%q(i,:,ixcldliq), &
#ifndef CRM3D
#ifdef SP_DIR_NS
               state(c)%q(i,:,ixcldice), state(c)%v(i,:), state(c)%u(i,:),&
#else
               state(c)%q(i,:,ixcldice), state(c)%u(i,:), state(c)%v(i,:),&
#endif
#endif
#ifdef CRM3D
               state(c)%q(i,:,ixcldice), state(c)%u(i,:), state(c)%v(i,:),&
#endif
               state(c)%ps(i), state(c)%pmid(i,:), state(c)%pdel(i,:), state(c)%phis(i), &
               state(c)%zm(i,:), state(c)%zi(i,:), ztodt, pver, &
               ptend(c)%q(i,:,1), ptend(c)%q(i,:,ixcldliq), ptend(c)%q(i,:,ixcldice), ptend(c)%s(i,:), &
! pritch adding capabilities for SPMOMTRANS:
               ptend(c)%u(i,:), ptend(c)%v(i,:), &
! end 
               u_crm(i,:,:,:,c), v_crm(i,:,:,:,c), w_crm(i,:,:,:,c), &
               t_crm(i,:,:,:,c), q_crm(i,:,:,:,c), qn_crm(i,:,:,:,c), qp_crm(i,:,:,:,c) ,&
               qc_crm(i,:,:,:,c), qi_crm(i,:,:,:,c), qpc_crm(i,:,:,:,c), qpi_crm(i,:,:,:,c),  &
               prec_crm(i,:,:,c), qrs_crm(i,:,:,:,c), qrl_crm(i,:,:,:,c), &
	       fsds_crm(i,:,:,c), fsns_crm(i,:,:,c), fsntoa_crm(i,:,:,c), fsutoa_crm(i,:,:,c),  &
	       flwds_crm(i,:,:,c), flns_crm(i,:,:,c), flut_crm(i,:,:,c),    &
	       fsntoac_crm(i,:,:,c), fsdsc_crm(i,:,:,c), flutc_crm(i,:,:,c), flnsc_crm(i,:,:,c), &
               t_rad(:,:,:,c), qv_rad(:,:,:,c), qc_rad(:,:,:,c), qi_rad(:,:,:,c), &
               precc(i,c), precl(i,c), precsc(i,c), precsl(i,c), &
               cltot(i,c), clhgh(i,c), clmed(i,c), cllow(i,c), &
               stat_buffer(i,:,c), spcld(i,:,c), cldr(i,:,c), cldtop(i,:,c) , &
	       gicewp(i,:,c), gliqwp(i,:,c),     &
               mctot(i,:,c), mcup(i,:,c), mcdn(i,:,c), mcuup(i,:,c), mcudn(i,:,c), &
               crm_qc(i,:,c),crm_qi(i,:,c),crm_qs(i,:,c),crm_qg(i,:,c),crm_qr(i,:,c), &
               tkez(i,:,c),tkesgsz(i,:,c),flux_u(i,:,c),flux_v(i,:,c),flux_qt(i,:,c),fluxsgs_qt(i,:,c),flux_qp(i,:,c), &
               precflux(i,:,c),qt_ls(i,:,c), qt_trans(i,:,c), qp_trans(i,:,c), qp_fall(i,:,c), &
               qp_evp(i,:,c), qp_src(i,:,c), t_ls(i,:,c), prectend(i,c), precstend(i,c), &
               nbreak(c), doinitial(c),  in_ocnfrac(i,c), wnd(c), tau00(c), bflx(i,c), taux_crm(i,c), tauy_crm(i,c), z0m(i,c), &
#ifdef CRMACCEL
               do_accel, crm_accel_factor, &
#endif               
               timing_factor(i,c),u_ls(i,:,c),v_ls(i,:,c) &
               )

   call t_stopf('crm_call')
!
! CRM column-by-column radiation calculations
!
!
! Loop over CRM columns:

          do jj=1,crm_ny
           do ii=1,crm_nx

!
!          Compute liquid and ice water paths for a given CRM column
            do m=1,crm_nz
               k = pver-m+1
               trad(i,k,c) = t_rad(ii,jj,m,c)
               qvrad(i,k,1,c) = qv_rad(ii,jj,m,c)
               qtot(c) = qc_rad(ii,jj,m,c) + qi_rad(ii,jj,m,c)
               if(qtot(c).gt.1.e-9) then
                 fice(i,k,c) = qi_rad(ii,jj,m,c)/qtot(c)
                 cldn(i,k,c) = 0.99_r8
		 cld_crm(ii,jj,m,c)=0.99_r8
                 cicewp(i,k,c) = qi_rad(ii,jj,m,c)*state(c)%pdel(i,k)/gravit*1000.0 &
		           / max(0.01_r8,cldn(i,k,c)) ! In-cloud ice water path.
                 cliqwp(i,k,c) = qc_rad(ii,jj,m,c)*state(c)%pdel(i,k)/gravit*1000.0 &
		           / max(0.01_r8,cldn(i,k,c)) ! In-cloud liquid water path.
               else
                 fice(i,k,c)=0.
                 cldn(i,k,c)=0.
		 cld_crm(ii,jj,m,c)=0.
                 cicewp(i,k,c) = 0.           ! In-cloud ice water path.
                 cliqwp(i,k,c) = 0.           ! In-cloud liquid water path.
               end if
               cwp(i,k,c) = cicewp(i,k,c) + cliqwp(i,k,c)
	       cliqwp_crm(ii,jj,m,c)=cliqwp(i,k,c)
	       cicewp_crm(ii,jj,m,c)=cicewp(i,k,c)
            end do

!           Cloud water and ice particle sizes
            call cldefr(c, i, i, in_landfrac(:,c), state(c)%t, rel(:,:,c), rei(:,:,c), state(c)%ps, state(c)%pmid, in_landm(:,c), in_icefrac(:,c), snowh(:,c))

!           Cloud emissivity.
            call cldems(lchnk, i, i, cwp(:,:,c), fice(:,:,c), rei(:,:,c), emis(:,:,c))

            if(doisccp)then
             do m=1,crm_nz
               k = pver-m+1
	       rel_crm(ii,jj,m,c)=rel(i,k,c)
	       rei_crm(ii,jj,m,c)=rei(i,k,c)
	       emis_crm(ii,jj,m,c)=emis(i,k,c)
             end do
	        endif
            call cldovrlap(lchnk,i,i,state(c)%pint,cldn(:,:,c),nmxrgn(:,c),pmxrgn(:,:,c))

!
!   Compute radiation:
   call t_startf ('crmrad_call')
            call radctl (lchnk, i, i, lwup(:,c), emis(:,:,c), state(c)%pmid, &
                state(c)%pint, state(c)%lnpmid, state(c)%lnpint, trad(:,:,c), qvrad(:,:,:,c), &
                cldn(:,:,c), cicewp(:,:,c), cliqwp(:,:,c), coszrs(:,c), asdir(:,c), asdif(:,c),               &
                aldir(:,c), aldif(:,c), pmxrgn(:,:,c), nmxrgn(:,c), &
                fsns_crm(1,ii,jj,c), fsnt_crm(1,ii,jj,c), flns_crm(1,ii,jj,c), flnt_crm(1,ii,jj,c), &
                qrs_tmp(:,:,c), qrl_tmp(:,:,c), flwds_crm(1,ii,jj,c), rel(:,:,c), rei(:,:,c),  &
                sols_crm(1,ii,jj,c), soll_crm(1,ii,jj,c), solsd_crm(1,ii,jj,c), solld_crm(1,ii,jj,c),  &
                 in_landfrac(:,c), state(c)%zm, state(c), fsds_crm(1,ii,jj,c), &
                fsntoa_crm(1,ii,jj,c)  ,fsntoac_crm(1,ii,jj,c) ,fsdsc_crm(1,ii,jj,c),   &
				flwdsc_crm(1,ii,jj,c),fsntc_crm(1,ii,jj,c) ,fsnsc_crm(1,ii,jj,c), &
                fsutoa_crm(1,ii,jj,c) ,fsutoac_crm(1,ii,jj,c) ,flut_crm(1,ii,jj,c) , &
				flutc_crm(1,ii,jj,c) ,flntc_crm(1,ii,jj,c) ,flnsc_crm(1,ii,jj,c) ,solin_crm(1,ii,jj,c), &
                .false., .false.,.true.,.true.)
   call t_stopf('crmrad_call')
		
            do m=1,crm_nz
               k = pver-m+1
               qrs_crm(i,ii,jj,m,c) = (qrs_tmp(i,k,c)) / cpair
               qrl_crm(i,ii,jj,m,c) = (qrl_tmp(i,k,c)) / cpair
            end do

            if(ii.eq.1.and.jj.eq.1) then
              stat_buffer(i,10,c) = stat_buffer(i,10,c) + fsds_crm(i,ii,jj,c)
              stat_buffer(i,11,c) = stat_buffer(i,11,c) + flwds_crm(i,ii,jj,c)
              stat_buffer(i,12,c) = stat_buffer(i,12,c) + fsutoa_crm(i,ii,jj,c)
              stat_buffer(i,13,c) = stat_buffer(i,13,c) + flut_crm(i,ii,jj,c)
              stat_buffer(i,14,c) = stat_buffer(i,14,c) + fsdsc_crm(i,ii,jj,c)
              stat_buffer(i,15,c) = stat_buffer(i,15,c) + fsutoac_crm(i,ii,jj,c)
            end if

            if(mm.ne.nbreak(c)) then

             qrs(i,:,c) = qrs(i,:,c) + qrs_tmp(i,:,c)  * state(c)%pdel(i,:)
             qrl(i,:,c) = qrl(i,:,c) + qrl_tmp(i,:,c)  * state(c)%pdel(i,:)
             solin(i,c) = solin(i,c) + solin_crm(i,ii,jj,c)
             in_fsnt(i,c) = in_fsnt(i,c) + fsnt_crm(i,ii,jj,c)
             in_fsns(i,c) = in_fsns(i,c) + fsns_crm(i,ii,jj,c)
             fsds(i,c) = fsds(i,c) + fsds_crm(i,ii,jj,c)
             fsdsc(i,c) = fsdsc(i,c) + fsdsc_crm(i,ii,jj,c)
             fsutoa(i,c) = fsutoa(i,c) + fsutoa_crm(i,ii,jj,c)
             fsutoac(i,c) = fsutoac(i,c) + fsutoac_crm(i,ii,jj,c)
             fsntc(i,c) = fsntc(i,c) + fsntc_crm(i,ii,jj,c)
             fsnsc(i,c) = fsnsc(i,c) + fsnsc_crm(i,ii,jj,c)
             fsntoa(i,c) = fsntoa(i,c) + fsntoa_crm(i,ii,jj,c)
             fsntoac(i,c) = fsntoac(i,c) + fsntoac_crm(i,ii,jj,c)
             sols(i,c) = sols(i,c) + sols_crm(i,ii,jj,c)
             soll(i,c) = soll(i,c) + soll_crm(i,ii,jj,c)
             solsd(i,c) = solsd(i,c) + solsd_crm(i,ii,jj,c)
             solld(i,c) = solld(i,c) + solld_crm(i,ii,jj,c)
             flut(i,c) = flut(i,c) + flut_crm(i,ii,jj,c)
             flutc(i,c) = flutc(i,c) + flutc_crm(i,ii,jj,c)
             in_flnt(i,c) = in_flnt(i,c) + flnt_crm(i,ii,jj,c)
             in_flns(i,c) = in_flns(i,c) + flns_crm(i,ii,jj,c)
             flntc(i,c) = flntc(i,c) + flntc_crm(i,ii,jj,c)
             flnsc(i,c) = flnsc(i,c) + flnsc_crm(i,ii,jj,c)
             flwds(i,c) = flwds(i,c) + flwds_crm(i,ii,jj,c)
             flwdsc(i,c) = flwdsc(i,c) + flwdsc_crm(i,ii,jj,c)

!
!   Save to the buffer for use in the next step for averaging:
!
            else

             qrs1(i,:,c) = qrs1(i,:,c) + qrs_tmp(i,:,c)
             qrl1(i,:,c) = qrl1(i,:,c) + qrl_tmp(i,:,c)
             in_rad_buffer(i,1,c) =  in_rad_buffer(i,1,c) + solin_crm(i,ii,jj,c)
             in_rad_buffer(i,2,c) =  in_rad_buffer(i,2,c) + fsds_crm(i,ii,jj,c)
             in_rad_buffer(i,3,c) =  in_rad_buffer(i,3,c) + fsdsc_crm(i,ii,jj,c)
             in_rad_buffer(i,4,c) =  in_rad_buffer(i,4,c) + fsnt_crm(i,ii,jj,c)
             in_rad_buffer(i,5,c) =  in_rad_buffer(i,5,c) + fsns_crm(i,ii,jj,c)
             in_rad_buffer(i,6,c) =  in_rad_buffer(i,6,c) + fsntc_crm(i,ii,jj,c)
             in_rad_buffer(i,7,c) =  in_rad_buffer(i,7,c) + fsnsc_crm(i,ii,jj,c)
             in_rad_buffer(i,8,c) =  in_rad_buffer(i,8,c) + fsntoa_crm(i,ii,jj,c)
             in_rad_buffer(i,9,c) =  in_rad_buffer(i,9,c) + fsntoac_crm(i,ii,jj,c)
             in_rad_buffer(i,10,c) = in_rad_buffer(i,10,c) + sols_crm(i,ii,jj,c)
             in_rad_buffer(i,11,c) = in_rad_buffer(i,11,c) + soll_crm(i,ii,jj,c)
             in_rad_buffer(i,12,c) = in_rad_buffer(i,12,c) + solsd_crm(i,ii,jj,c)
             in_rad_buffer(i,13,c) = in_rad_buffer(i,13,c) + solld_crm(i,ii,jj,c)
             in_rad_buffer(i,14,c) = in_rad_buffer(i,14,c) + flnt_crm(i,ii,jj,c)
             in_rad_buffer(i,15,c) = in_rad_buffer(i,15,c) + flut_crm(i,ii,jj,c)
             in_rad_buffer(i,16,c) = in_rad_buffer(i,16,c) + flntc_crm(i,ii,jj,c)
             in_rad_buffer(i,17,c) = in_rad_buffer(i,17,c) + flutc_crm(i,ii,jj,c)
             in_rad_buffer(i,18,c) = in_rad_buffer(i,18,c) + flns_crm(i,ii,jj,c)
             in_rad_buffer(i,19,c) = in_rad_buffer(i,19,c) + flnsc_crm(i,ii,jj,c)
             in_rad_buffer(i,20,c) = in_rad_buffer(i,20,c) + flwds_crm(i,ii,jj,c)
             in_rad_buffer(i,21,c) = in_rad_buffer(i,21,c) + fsutoa_crm(i,ii,jj,c)
             in_rad_buffer(i,22,c) = in_rad_buffer(i,22,c) + fsutoac_crm(i,ii,jj,c)
             in_rad_buffer(i,23,c) = in_rad_buffer(i,23,c) + flwdsc_crm(i,ii,jj,c)

           end if

          end do ! ii
         end do ! jj

        end do ! mm
          if (doisccp) then
            call crm_isccp (state(c)%pmid(i,:), state(c)%pint(i,:), qv_rad(:,:,:,c), t_rad(:,:,:,c), ts(i,c),    &
 cld_crm(:,:,:,c), cliqwp_crm(:,:,:,c), cicewp_crm(:,:,:,c), rel_crm(:,:,:,c), rei_crm(:,:,:,c), emis_crm(:,:,:,c), coszrs(i,c),    &
	        fq_isccp_s1(i,:,c), totalcldarea(i,c), lowcldarea(i,c), midcldarea(i,c), hghcldarea(i,c), &
                meantaucld(i,c), meanptop(i,c), meanttop(i,c), cloudy(i,c))               
        end if

        qrs1(i,:,c) = qrs1(i,:,c) * state(c)%pdel(i,:)
        qrl1(i,:,c) = qrl1(i,:,c) * state(c)%pdel(i,:)
        coef(c) = 1._r8/dble(nbreak(c)*crm_nx*crm_ny)
        qrs(i,:,c) = qrs(i,:,c) * coef(c) / state(c)%pdel(i,:)
        qrl(i,:,c) = qrl(i,:,c) * coef(c) / state(c)%pdel(i,:)
        solin(i,c) = solin(i,c) * coef(c)
        in_fsnt(i,c) = in_fsnt(i,c) * coef(c)
        in_fsns(i,c) = in_fsns(i,c) * coef(c)
        fsntoa(i,c) = fsntoa(i,c) * coef(c)
        fsntoac(i,c) = fsntoac(i,c) * coef(c)
        fsds(i,c) = fsds(i,c) * coef(c)
        fsdsc(i,c) = fsdsc(i,c) * coef(c)
        fsutoa(i,c) = fsutoa(i,c) * coef(c)
        fsutoac(i,c) = fsutoac(i,c) * coef(c)
        fsntc(i,c) = fsntc(i,c) * coef(c)
        fsnsc(i,c) = fsnsc(i,c) * coef(c)
        sols(i,c) = sols(i,c) * coef(c)
        soll(i,c) = soll(i,c) * coef(c)
        solsd(i,c) = solsd(i,c) * coef(c)
        solld(i,c) = solld(i,c) * coef(c)
        flut(i,c) = flut(i,c) * coef(c)
        flutc(i,c) = flutc(i,c) * coef(c)
        flntc(i,c) = flntc(i,c) * coef(c)
        flnsc(i,c) = flnsc(i,c) * coef(c)
        in_flnt(i,c) = in_flnt(i,c) * coef(c)
        in_flns(i,c) = in_flns(i,c) * coef(c)
        flwds(i,c) = flwds(i,c) * coef(c)
        flwdsc(i,c) = flwdsc(i,c) * coef(c)
        lwcf(i,c) = flutc(i,c) - flut(i,c)
        swcf(i,c) = fsntoa(i,c) - fsntoac(i,c)
        stat_buffer(i,10:15,c) = stat_buffer(i,10:15,c)/dble(nbreak(c))

     end do ! i

 end do ! ---------- PRITCH END CRITICAL CHUNK LOOP BRACKETING MOST WORK, 
     ! ( and minimal external libraries/ functions, optimizes MIC-compatibility).
     
 do c=begchunk,endchunk ! ---- pritch new chunk loop -------------------

   lchnk = state(c)%lchnk
   ncol  = state(c)%ncol 
   ifld = pbuf_get_fld_idx('CLD')
   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   cld(:,:) = spcld(:,:,c)
   
     if (doisccp) then
        if (any(coszrs(:ncol,c) > 0.)) then
           call outfld('FISCCP1 ',fq_isccp_s1(:,:,c), pcols,lchnk)
           call outfld('TCLDAREA',totalcldarea(:,c),pcols,lchnk)
           call outfld('LCLDAREA',lowcldarea(:,c),pcols,lchnk)
           call outfld('MCLDAREA',midcldarea(:,c),pcols,lchnk)
           call outfld('HCLDAREA',hghcldarea(:,c),pcols,lchnk)
           call outfld('MEANPTOP',meanptop(:,c)    ,pcols,lchnk)
           call outfld('MEANTAU ',meantaucld(:,c)  ,pcols,lchnk)
           call outfld('MEANTTOP',meanttop(:,c)    ,pcols,lchnk)
           call outfld('CLOUDY  ',cloudy(:,c)      ,pcols,lchnk)
        end if
     end if
!
!  subtract radiative heating tendency from the CRM tendency:
!  it will be added later:

     ptend(c)%s(:ncol,pver-crm_nz+1:pver) = ptend(c)%s(:ncol,pver-crm_nz+1:pver) -  &
        (qrs(:ncol,pver-crm_nz+1:pver,c) + qrl(:ncol,pver-crm_nz+1:pver,c))

     ptend(c)%name  = 'crm'
     ptend(c)%ls    = .TRUE.
     ptend(c)%lq(1) = .TRUE.
     ptend(c)%lq(ixcldliq) = .TRUE.
     ptend(c)%lq(ixcldice) = .TRUE.
! below, pritch updated previous #ifdef CRM3D (which was linked to zero ptend%u)
! to this #ifdef SPMOMTRANS, now linked to CRM-computed ptend%u
#ifdef SPMOMTRANS 
   ptend(c)%lu    = .TRUE.
   ptend(c)%lv    = .TRUE.
   call outfld('UCONVMOM',ptend(c)%u,pcols   ,lchnk   )
   call outfld('VCONVMOM',ptend(c)%v,pcols   ,lchnk   )
#endif
   call outfld('PRES    ',state(c)%pmid ,pcols   ,lchnk   )
   call outfld('DPRES   ',state(c)%pdel ,pcols   ,lchnk   )
   call outfld('HEIGHT  ',state(c)%zm   ,pcols   ,lchnk   )
   call outfld('CRM_U   ',u_crm(:,:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_V   ',v_crm(:,:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_W   ',w_crm(:,:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_TABS',t_crm(:,:,:,:,c),pcols   ,lchnk   )

   call outfld('CRM_QV  ',(q_crm(:,:,:,:,c)-qc_crm(:,:,:,:,c)-qi_crm(:,:,:,:,c))*1000.,pcols   ,lchnk   )
   call outfld('CRM_QC  ',qc_crm(:,:,:,:,c)*1000.   ,pcols   ,lchnk   )
   call outfld('CRM_QI  ',qi_crm(:,:,:,:,c)*1000.   ,pcols   ,lchnk   )
   call outfld('CRM_QPC ',qpc_crm(:,:,:,:,c)*1000.  ,pcols   ,lchnk   )
   call outfld('CRM_QPI ',qpi_crm(:,:,:,:,c)*1000.  ,pcols   ,lchnk   )
   call outfld('CRM_PREC',prec_crm(:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_QRS ',qrs_crm(:,:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_QRL ',qrl_crm(:,:,:,:,c),pcols   ,lchnk   )
   call outfld('CRM_FSNT',fsntoa_crm(:,:,:,c),pcols   ,lchnk   )
   call outfld('CRMFSNTC',fsntoac_crm(:,:,:,c)    ,pcols   ,lchnk   )
   call outfld('CRM_FSUT',fsutoa_crm (:,:,:,c)    ,pcols   ,lchnk   )
   call outfld('CRM_FSNS',fsns_crm  (:,:,:,c)     ,pcols   ,lchnk   )
   call outfld('CRMFSDSC',fsdsc_crm (:,:,:,c)     ,pcols   ,lchnk   )
   call outfld('CRM_FSDS',fsds_crm (:,:,:,c)      ,pcols   ,lchnk   )
   call outfld('CRM_FLUT',flut_crm (:,:,:,c)      ,pcols   ,lchnk   )
   call outfld('CRMFLUTC',flutc_crm(:,:,:,c)      ,pcols   ,lchnk   )
   call outfld('CRM_FLNS',flns_crm (:,:,:,c)      ,pcols   ,lchnk   )
   call outfld('CRMFLNSC',flnsc_crm(:,:,:,c)      ,pcols   ,lchnk   )
   call outfld('CRM_FLDS',flwds_crm(:,:,:,c)      ,pcols   ,lchnk   )

   ftem(:ncol,:pver,c) = ptend(c)%s(:ncol,:pver)/cpair
   call outfld('SPDT    ',ftem(:,:,c),pcols   ,lchnk   )
   call outfld('SPDQ    ',ptend(c)%q(1,1,1) ,pcols   ,lchnk   )
   call outfld('SPDQC   ',ptend(c)%q(1,1,ixcldliq) ,pcols   ,lchnk   )
   call outfld('SPDQI   ',ptend(c)%q(1,1,ixcldice) ,pcols   ,lchnk   )

   call outfld('SPMC    ',mctot(:,:,c)     ,pcols   ,lchnk   )
   call outfld('SPMCUP  ',mcup (:,:,c)          ,pcols   ,lchnk   )
   call outfld('SPMCDN  ',mcdn  (:,:,c)         ,pcols   ,lchnk   )
   call outfld('SPMCUUP ',mcuup (:,:,c)         ,pcols   ,lchnk   )
   call outfld('SPMCUDN ',mcudn (:,:,c)         ,pcols   ,lchnk   )
   call outfld('SPQC    ',crm_qc (:,:,c)        ,pcols   ,lchnk   )
   call outfld('SPQI    ',crm_qi  (:,:,c)       ,pcols   ,lchnk   )
   call outfld('SPQS    ',crm_qs (:,:,c)        ,pcols   ,lchnk   )
   call outfld('SPQG    ',crm_qg (:,:,c)        ,pcols   ,lchnk   )
   call outfld('SPQR    ',crm_qr  (:,:,c)       ,pcols   ,lchnk   )

   call outfld('SPQTFLX ',flux_qt (:,:,c)       ,pcols   ,lchnk   )
   call outfld('SPUFLX  ',flux_u  (:,:,c)       ,pcols   ,lchnk   )
   call outfld('SPVFLX  ',flux_v  (:,:,c)       ,pcols   ,lchnk   )
   call outfld('TKE     ',tkez  (:,:,c)         ,pcols   ,lchnk   )
   call outfld('TKES    ',tkesgsz  (:,:,c)      ,pcols   ,lchnk   )
   call outfld('SPQTFLXS',fluxsgs_qt (:,:,c)    ,pcols   ,lchnk   )
   call outfld('SPQPFLX ',flux_qp  (:,:,c)      ,pcols   ,lchnk   )
   call outfld('SPPFLX  ',precflux (:,:,c)      ,pcols   ,lchnk   )
   call outfld('SPQTLS  ',qt_ls  (:,:,c)        ,pcols   ,lchnk   )
   call outfld('SPQTTR  ',qt_trans (:,:,c)      ,pcols   ,lchnk   )
   call outfld('SPQPTR  ',qp_trans (:,:,c)      ,pcols   ,lchnk   )
   call outfld('SPQPEVP ',qp_evp (:,:,c)        ,pcols   ,lchnk   )
   call outfld('SPQPFALL',qp_fall (:,:,c)       ,pcols   ,lchnk   )
   call outfld('SPQPSRC ',qp_src  (:,:,c)       ,pcols   ,lchnk   )
   call outfld('SPTLS   ',t_ls (:,:,c)          ,pcols   ,lchnk   )
   call outfld ('SPULS   ',u_ls(:,:,c), pcols,lchnk)
   call outfld ('SPVLS   ',v_ls(:,:,c), pcols,lchnk)

   call outfld('CLOUD   ',cld(:,:),  pcols,lchnk)
   call outfld('CLOUDR  ',cldr(:,:,c),  pcols,lchnk)
   call outfld('CLOUDTOP',cldtop(:,:,c), pcols,lchnk)
   call outfld('CLDTOT  ',cltot(:,c)  ,pcols,lchnk)
   call outfld('CLDHGH  ',clhgh(:,c)  ,pcols,lchnk)
   call outfld('CLDMED  ',clmed(:,c)  ,pcols,lchnk)
   call outfld('CLDLOW  ',cllow(:,c)  ,pcols,lchnk)

   call outfld('CLDLOWR ',stat_buffer(:,1,c),pcols,lchnk)
   call outfld('CLDMEDR ',stat_buffer(:,2,c),pcols,lchnk)
   call outfld('CLDHGHR ',stat_buffer(:,3,c),pcols,lchnk)
   call outfld('CLDTOTR ',stat_buffer(:,4,c),pcols,lchnk)
   call outfld('LWPR    ',stat_buffer(:,5,c),pcols,lchnk)
   call outfld('IWPR    ',stat_buffer(:,6,c),pcols,lchnk)
   call outfld('LOBR    ',stat_buffer(:,7,c),pcols,lchnk)
   call outfld('HIBR    ',stat_buffer(:,8,c),pcols,lchnk)
   call outfld('PRECR   ',stat_buffer(:,9,c),pcols,lchnk)
   call outfld('FSDSR   ',stat_buffer(:,10,c),pcols,lchnk)
   call outfld('FLDSR   ',stat_buffer(:,11,c),pcols,lchnk)
   call outfld('FSUTR   ',stat_buffer(:,12,c),pcols,lchnk)
   call outfld('FLUTR   ',stat_buffer(:,13,c),pcols,lchnk)
   call outfld('FSDSCR  ',stat_buffer(:,14,c),pcols,lchnk)
   call outfld('FSUTCR  ',stat_buffer(:,15,c),pcols,lchnk)
   call outfld('CLDLOWRD',stat_buffer(:,16,c),pcols,lchnk)
   call outfld('CLDMEDRD',stat_buffer(:,17,c),pcols,lchnk)
   call outfld('CLDHGHRD',stat_buffer(:,18,c),pcols,lchnk)
   call outfld('CLDTOTRD',stat_buffer(:,19,c),pcols,lchnk)

   call outfld('QRS     ',qrs(:,:,c)/cpair  ,pcols,lchnk)
   call outfld('SOLIN   ',solin(:,c) ,pcols,lchnk)
   call outfld('FSDS    ',fsds(:,c)  ,pcols,lchnk)
   call outfld('FSDSC   ',fsdsc(:,c) ,pcols,lchnk)
   call outfld('FSUTOA  ',fsutoa(:,c),pcols,lchnk)
   call outfld('FSUTOAC ',fsutoac(:,c),pcols,lchnk)
   call outfld('FSNT    ',in_fsnt(:,c)  ,pcols,lchnk)
   call outfld('FSNS    ',in_fsns(:,c)  ,pcols,lchnk)
   call outfld('FSNTC   ',fsntc(:,c) ,pcols,lchnk)
   call outfld('FSNSC   ',fsnsc(:,c) ,pcols,lchnk)
   call outfld('FSNTOA  ',fsntoa(:,c),pcols,lchnk)
   call outfld('FSNTOAC ',fsntoac(:,c),pcols,lchnk)
   call outfld('SOLS    ',sols(:,c)  ,pcols,lchnk)
   call outfld('SOLL    ',soll(:,c)  ,pcols,lchnk)
   call outfld('SOLSD   ',solsd(:,c) ,pcols,lchnk)
   call outfld('SOLLD   ',solld(:,c) ,pcols,lchnk)
   call outfld('QRL     ',qrl(:,:,c)/cpair ,pcols,lchnk)
   call outfld('FLNT    ',in_flnt(:,c)  ,pcols,lchnk)
   call outfld('FLUT    ',flut(:,c)  ,pcols,lchnk)
   call outfld('FLUTC   ',flutc(:,c) ,pcols,lchnk)
   call outfld('FLNTC   ',flntc(:,c) ,pcols,lchnk)
   call outfld('FLNS    ',in_flns(:,c)  ,pcols,lchnk)
   call outfld('FLNSC   ',flnsc(:,c) ,pcols,lchnk)
   call outfld('FLWDS   ',flwds(:,c) ,pcols,lchnk)
   call outfld('FLWDSC  ',flwdsc(:,c),pcols,lchnk)
   call outfld('LWCF    ',lwcf(:,c)  ,pcols,lchnk)
   call outfld('SWCF    ',swcf(:,c)  ,pcols,lchnk)

   call outfld('Z0M     ',z0m(:,c)  ,pcols,lchnk)
   call outfld('TAUX_CRM',taux_crm(:,c)  ,pcols,lchnk)
   call outfld('TAUY_CRM',tauy_crm(:,c)  ,pcols,lchnk)

   call outfld ('BFLX',bflx(:,c),pcols,lchnk)

   call outfld('TIMINGF ',timing_factor(:,c),pcols,lchnk)

! Compute energy and water integrals of input state

     call check_energy_timestep_init(state(c), tend(c), pbuf)
     call physics_update (state(c), tend(c), ptend(c), ztodt)
!    check energy integrals
     wtricesink(:ncol,c) = precc(:ncol,c) + precl(:ncol,c) + prectend(:ncol,c)*1.e-3 ! include precip storage term
     icesink(:ncol,c) = precsc(:ncol,c) + precsl(:ncol,c) + precstend(:ncol,c)*1.e-3   ! conversion of ice to snow
!     write(6,'(a,12e10.3)')'prect=',(prect(i),i=1,12)
     call check_energy_chng(state(c), tend(c), "crm", nstep, ztodt, zero, wtricesink(:,c), icesink(:,c), zero)

!
! Compute liquid water paths (for diagnostics only)
!
    tgicewp(:ncol,c) = 0.
    tgliqwp(:ncol,c) = 0.
    do k=1,pver
       do i = 1,ncol
          cicewp(i,k,c) = gicewp(i,k,c) / max(0.01_r8,cld(i,k)) ! In-cloud ice water path.
          cliqwp(i,k,c) = gliqwp(i,k,c) / max(0.01_r8,cld(i,k)) ! In-cloud liquid water path.
          tgicewp(i,c)  = tgicewp(i,c) + gicewp(i,k,c) ! grid cell mean ice water path.
          tgliqwp(i,c)  = tgliqwp(i,c) + gliqwp(i,k,c) ! grid cell mean ice water path.
       end do
    end do
    ! INSERT redefine pointers in each scope. 
    
    tgwp(:ncol,c) = tgicewp(:ncol,c) + tgliqwp(:ncol,c)
    gwp(:ncol,:pver,c) = gicewp(:ncol,:pver,c) + gliqwp(:ncol,:pver,c)
    cwp(:ncol,:pver,c) = cicewp(:ncol,:pver,c) + cliqwp(:ncol,:pver,c)

    call outfld('GCLDLWP' ,gwp(:,:,c)    , pcols,lchnk)
    call outfld('TGCLDCWP',tgwp(:,c)   , pcols,lchnk)
    call outfld('TGCLDLWP',tgliqwp(:,c), pcols,lchnk)
    call outfld('TGCLDIWP',tgicewp(:,c), pcols,lchnk)
    call outfld('ICLDLWP' ,cwp(:,:,c)    , pcols,lchnk)

   end do ! pritch chunk loop
   end if ! (is_first_step())

  do c=begchunk,endchunk
   lchnk = state(c)%lchnk
   ncol  = state(c)%ncol 
   ifld = pbuf_get_fld_idx('CLD')
   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

   do m=1,crm_nz
      k = pver-m+1
      do i=1,ncol
         qrs_crm(i,:,:,m,c) = qrs_crm(i,:,:,m,c) * state(c)%pdel(i,k) ! for energy conservation
         qrl_crm(i,:,:,m,c) = qrl_crm(i,:,:,m,c) * state(c)%pdel(i,k) ! for energy conservation
   end do
   end do



   call diag_dynvar (lchnk, ncol, state(c))

!========================================================
!========================================================
!========================================================
! End of superparameterization zone.
  end do ! end pritch new chunk loop
  call t_stopf('crm')
#endif
 do c=begchunk,endchunk ! pritch new chunk loop
   lchnk = state(c)%lchnk
   ncol  = state(c)%ncol 
   ifld = pbuf_get_fld_idx('CLD')
   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)


  if(l_analyses) then ! ONly do nudging if bndtva was specified in the namelist

      ptend(c)%name  = "analyses_nudge"
      s_tmp(:ncol,:pver,c) = state(c)%s(:ncol,:pver)
     if (nudge_dse_not_T) then
       call outfld('MSPSA',s_a,pcols,lchnk)
     else
       do k=1,pver
         aux_s_a (:ncol,k,c) = cpair*t_a(:ncol,k,lchnk) + gravit*state(c)%zm(:ncol,k) + state(c)%phis(:ncol)
       end do
       call outfld('MSPSA',aux_s_a(:,:,c),pcols,lchnk)
     endif
     call outfld('MSPS',s_tmp(:,:,c),pcols,lchnk)

     if (nudge_dse_not_T) then
      call analyses_nudge(ztodt   ,tau_t   ,ncol   ,pver    ,s_a(1,1,lchnk), &
                         state(c)%s          ,ptend(c)%s       ,ptend(c)%ls)
     else
      call analyses_nudge(ztodt   ,tau_t   ,ncol   ,pver    ,aux_s_a(:,:,c)  , &
                         state(c)%s          ,ptend(c)%s       ,ptend(c)%ls)
     end if
      call analyses_nudge(ztodt   ,tau_u   ,ncol   ,pver    ,u_a (1,1,lchnk), &
                         state(c)%u          ,ptend(c)%u       ,ptend(c)%lu)
      call analyses_nudge(ztodt   ,tau_v   ,ncol   ,pver    ,v_a (1,1,lchnk), &
                         state(c)%v          ,ptend(c)%v       ,ptend(c)%lv)
      call analyses_nudge(ztodt   ,tau_q   ,ncol   ,pver    ,q_a (1,1,lchnk), &
                          state(c)%q(1,1,1)   ,ptend(c)%q(1,1,1),ptend(c)%lq(1))

      ps_local(:ncol,c) =      state(c)%ps(:ncol)
      state(c)%ps(:ncol) = log( state(c)%ps(:ncol) )
      call analyses_nudge(ztodt   ,tau_ps  ,ncol   ,1       ,ps_a(1,  lchnk), &
                          state(c)%ps         ,dpdt(:,c)       ,lpsntenl   )
      state(c)%ps(:ncol) = state(c)%ps(:ncol) + dpdt(:ncol,c) * ztodt
      state(c)%ps(:ncol) = exp( state(c)%ps(:ncol) )
      ps_local(:ncol,c) = ( state(c)%ps(:ncol) - ps_local(:ncol,c) )/ztodt

  end if

     call outfld('VNTEND  ',ptend(c)%v       ,pcols   ,lchnk   )
      call outfld('UNTEND  ',ptend(c)%u       ,pcols   ,lchnk   )
      call outfld('QNTEND  ',ptend(c)%q(1,1,1),pcols   ,lchnk   )
      call outfld('LPSNTEN ',ps_local(:,c)      ,pcols   ,lchnk   )
!
! pressure arrays
!
      call plevs0(ncol, pcols, pver, state(c)%ps,   state(c)%pint, &
                  state(c)%pmid, state(c)%pdel)
!
! log(pressure) arrays and Exner function
!
      state(c)%lnpint(:ncol,:pver+1) = log(state(c)%pint(:ncol,:pver+1))
      state(c)%rpdel (:ncol,:pver) = 1./state(c)%pdel (:ncol,:pver)
      state(c)%lnpmid(:ncol,:pver) = log(state(c)%pmid(:ncol,:pver))
      do k=1,pver
         do i=1,ncol
            state(c)%exner (i,k) = (state(c)%pint(i,pver+1) / state(c)%pmid(i,k))**cappa
         end do
      end do

      call physics_update(state(c), tend(c), ptend(c), ztodt)

      s_tmp(:ncol,:pver,c) = (state(c)%s(:ncol,:pver) - s_tmp(:ncol,:pver,c))/ztodt
      call outfld('SNTEND  ',s_tmp   ,pcols   ,lchnk   )

!
! Compute net flux
! Since in_fsns, in_fsnt, flns, and in_flnt are in the buffer, array values will be carried across
! timesteps when the radiation code is not invoked.
!
   do i=1,ncol
      tend(c)%flx_net(i) = in_fsnt(i,c) - in_fsns(i,c) - in_flnt(i,c) + in_flns(i,c)
   end do
!
! Compute net radiative heating
!
   call radheat_net (state(c), ptend(c), qrl(:,:,c), qrs(:,:,c))
!
! Add radiation tendencies to cummulative model tendencies and update profiles
!
   call physics_update(state(c), tend(c), ptend(c), ztodt)

! check energy integrals
   call check_energy_chng(state(c), tend(c), "radheat", nstep, ztodt, zero, zero, zero, tend(c)%flx_net)
!
! Compute net surface radiative flux for use by surface temperature code.
! Note that units have already been converted to mks in RADCTL.  Since
! fsns and flwds are in the buffer, array values will be carried across
! timesteps when the radiation code is not invoked.
!
   srfrad(:ncol,c) = in_fsns(:ncol,c) + flwds(:ncol,c)
   call outfld('SRFRAD  ',srfrad(:,c),pcols,lchnk)
!
! Save atmospheric fields to force surface models
!
   call srfxfer (lchnk, ncol, state(c)%ps, state(c)%u(1,pver), state(c)%v(1,pver),    &
                 state(c)%t(1,pver), state(c)%q(1,pver,1), state(c)%exner(1,pver), state(c)%zm(1,pver), &
                    state(c)%pmid,      &
                 state(c)%rpdel(1,pver))

!---------------------------------------------------------------------------------------
! Save history variables. These should move to the appropriate parameterization interface
!---------------------------------------------------------------------------------------

   call outfld('PRECL   ',precl(:,c)   ,pcols   ,lchnk       )
   call outfld('PRECC   ',precc(:,c)   ,pcols   ,lchnk       )
   call outfld('PRECSL  ',precsl(:,c)  ,pcols   ,lchnk       )
   call outfld('PRECSC  ',precsc(:,c)  ,pcols   ,lchnk       )
   
   prect(:ncol,c) = precc(:ncol,c) + precl(:ncol,c)
   call outfld('PRECT   ',prect(:,c)   ,pcols   ,lchnk       )
   call outfld('PRECTMX ',prect(:,c)   ,pcols   ,lchnk       )

#if ( defined COUP_CSM )
   call outfld('PRECLav ',precl(:,c)   ,pcols   ,lchnk   )
   call outfld('PRECCav ',precc(:,c)   ,pcols   ,lchnk   )
#endif
!     
! Compute heating rate for dtheta/dt
!
   do k=1,pver
      do i=1,ncol
         ftem(i,k,c) = (qrs(i,k,c) + qrl(i,k,c))/cpair * (1.e5/state(c)%pmid(i,k))**cappa
      end do
   end do
   call outfld('HR      ',ftem(:,:,c)   ,pcols   ,lchnk   )

   
#ifdef QRLDAMP
   call accumulate_dailymean_qrl(lchnk,ncol,qrl(:,:,c),nstep)
 ! pritch: artificial amplification/ damping of radiative
 ! heating anomalies with respect to an offline mean annual cycle. 
   call qrl_interference(lchnk,ncol,qrl(:,:,c),clat,newqrl(:,:,c),qrldampfac,qrldamp_equatoronly, qrl_dylat, qrl_critlat_deg, qrl_dailymean_interference, qrldamp_freetroponly, qrl_pbot,qrl_ptop,qrl_dp,state(c)%pmid)
   ptend(c)%name = 'qrldamp'
   do i=1,ncol
     do k=1,pver
       dqrl(i,k,c) = newqrl(i,k,c) - qrl(i,k,c)
     end do
   end do
!   if (masterproc) then
!     write (6,*) '==========='
!     write (6,*) 'newqrl(icol=1,:) = ',newqrl(1,:)
!     write (6,*) 'qrl(icol=1,:) = ', qrl(1,:)
!   end if

   ptend(c)%s(:ncol,:) = dqrl
   ptend(c)%ls = .true.
! Add artificial interference tendencies to cumulative model tendencies, update
! profiles:
   call physics_update(state(c),tend(c),ptend(c),ztodt)    
   call outfld('DQRL    ',dqrl(:,:,c)/cpair ,pcols,lchnk)
! Also apply artificial interference to the radiative heating driving the CRM
! on subsequent call to crm, identically in each of the CRM sub-columns...

 ! Now it appears to me that qrl_crm has units of qrl_tmp / cpair * pdel
 ! at this scope in the code whereas qrl has units of qrl_tmp
 ! Hence applying "qrl"-scope anomalies (dqrl) to qrl_crm variable means multiplying first by pdel / cpair.
   do jj=1,crm_ny
     do ii=1,crm_nx
       do m=1,crm_nz
         k = pver-m+1
         qrl_crm(i,ii,jj,m,c) = qrl_crm(i,ii,jj,m,c) + dqrl(i,k,c)/cpair*state(c)%pdel(i,k)
       end do
     end do
   end do 

#endif

! convert radiative heating rates to Q*dp for energy conservation
   if (conserve_energy) then
      do k =1 , pver
         do i = 1, ncol
            qrs(i,k,c) = qrs(i,k,c)*state(c)%pdel(i,k)
            qrl(i,k,c) = qrl(i,k,c)*state(c)%pdel(i,k)
         end do
      end do
   end if

     ! PRITCH OUTPUT REVERSE MAPPING (for all "srfflx" or "surface_state" variables specified 
     ! as (inout) or as (out) in the original scaffolding of tphysbc
     ! (which had to be shunted to local variables to avoid structure packaging). 
     
  in_surface_state2d(c)%precl  =  precl(:,c) 		
  in_surface_state2d(c)%precc  =  precc(:,c) 		
  in_surface_state2d(c)%precsl  =  precsl(:,c)    
  in_surface_state2d(c)%precsc       =  precsc(:,c)    
  in_surface_state2d(c)%flwds(:)  =  flwds(:,c) 		
  in_srfflx_state2d(c)%lwup(:)    =  lwup(:,c) 	      
  in_surface_state2d(c)%srfrad(:)  =  srfrad(:,c)    
  in_surface_state2d(c)%sols(:)   =  sols(:,c) 	 	
  in_surface_state2d(c)%soll(:)   =  soll(:,c) 	 	
  in_surface_state2d(c)%solsd(:)  =  solsd(:,c)	 	
  in_surface_state2d(c)%solld(:)      =  solld(:,c) 		      
end do ! PRITCH FINAL CHUNK (should be no need to thread it).

   return
end subroutine tphysbc_internallythreaded
