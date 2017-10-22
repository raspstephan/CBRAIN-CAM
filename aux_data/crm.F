
        subroutine crm (lchnk, icol, 
     &                  tl, ql, qcl, qci, ul, vl,
     &                  ps, pmid, pdel, phis, 
     &                  zmid, zint, dt_gl, plev,
     &                  qltend, qcltend, qcitend, sltend,
     &                  ultend,vltend,
     &                  u_crm, v_crm, w_crm, 
     &                  t_crm, q_crm, qn_crm, qp_crm,
     &                  qc_crm, qi_crm, qpc_crm, qpi_crm,
     &                  prec_crm, qrs_crm, qrl_crm,
     &	                fsds, fsns, fsnt, fsut,
     &	                flwds, flns, flut, 
     &                  fsntc, fsdsc, flutc, flnsc,
     &                  t_rad, qv_rad, qc_rad, qi_rad,
     &                  precc, precl, precsc, precsl, 
     &                  cltot, clhgh, clmed, cllow, 
     &                  stat_buffer, cld, cldr, cldtop, 
     &                  gicewp, gliqwp, 
     &                  mc, mcup, mcdn, mcuup, mcudn,
     &                  crm_qc, crm_qi, crm_qs, crm_qg, crm_qr,
     &                  tkez, tkesgsz, flux_u, flux_v, flux_qt, fluxsgs_qt, flux_qp, 
     &                  pflx, qt_ls, qt_trans, qp_trans, qp_fall, 
     &                  qp_evp, qp_src, t_ls, prectend, precstend,
     &                  nbreak, doinitial, ocnfrac, wnd, tau00, bflxls, 
     &                  taux_crm, tauy_crm, z0m, 
#ifdef CRMACCEL
     &                  do_accel,crm_accel_factor,
#endif
     &                  timing_factor, u_ls, v_ls
     &                  )
!---------------------------------------------------------------
!  Cloud resolving model's main driver (Marat Khairoutdinov)
!---------------------------------------------------------------

        use shr_kind_mod, only: r8 => shr_kind_r8
	use physconst, only: gravit, latvap, latice, cpair
	implicit none
	include 'common.inc'
	include 'params.inc'

!  Input:
	
         integer, intent(in) :: lchnk    ! chunk identifier
         integer, intent(in) :: icol     ! column identifier
         integer, intent(in) :: plev     ! number of levels
	 real(r8), intent(in) :: ps ! Global grid surface pressure (Pa)
	 real(r8), intent(in) :: pmid(plev) ! Global grid pressure (Pa)
	 real(r8), intent(in) :: pdel(plev) ! Layer's pressure thickness (Pa)
	 real(r8), intent(in) :: phis ! Global grid surface geopotential (m2/s2)
	 real(r8), intent(in) :: zmid(plev) ! Global grid height (m)
	 real(r8), intent(in) :: zint(plev+1)! Global grid interface height (m)
	 real(r8), intent(in) :: dt_gl ! global model's time step	
	 real(r8), intent(in) :: qrs_crm(crm_nx, crm_ny, crm_nz) ! CRM SW rad. heating
	 real(r8), intent(in) :: qrl_crm(crm_nx, crm_ny, crm_nz) ! CRM LW rad. heating
         real(r8), intent(in) :: fsds(crm_nx,crm_ny)   ! downward solar flux at surface
         real(r8), intent(in) :: fsns(crm_nx,crm_ny)   ! Surface solar absorbed flux
         real(r8), intent(in) :: fsdsc(crm_nx,crm_ny)  ! Clearsky downward solar flux at surface
         real(r8), intent(in) :: fsnt(crm_nx,crm_ny)   ! Net downard solar flux at top of atmosphere
         real(r8), intent(in) :: fsntc(crm_nx,crm_ny)  ! Clearsky net solar flux at TOA
         real(r8), intent(in) :: fsut(crm_nx,crm_ny)   ! Upwelling Shortwave Flux at TOA
         real(r8), intent(in) :: flwds(crm_nx,crm_ny)  ! Surface longwave down flux
         real(r8), intent(in) :: flns(crm_nx,crm_ny)   ! Srf longwave cooling (up-down) flux
         real(r8), intent(in) :: flnsc(crm_nx,crm_ny)  ! Clearsky srf longwave cooling (up-down) flux
         real(r8), intent(in) :: flut(crm_nx,crm_ny)   ! Outgoing lw flux at model top
         real(r8), intent(in) :: flutc(crm_nx,crm_ny)  ! Clearsky outgoing lw flux at model top
         integer, intent(in)  :: nbreak ! number of subcyclings
         real(r8), intent(in) :: ocnfrac ! area fraction of the ocean
         real(r8), intent(in) :: tau00  ! large-scale surface stress (N/m2)
         real(r8), intent(in) :: wnd  ! large-scale surface wind (m/s)
         real(r8), intent(in) :: bflxls  ! large-scale surface buoyancy flux (K m/s)
	 

!  Input/Output:

	 real(r8), intent(inout) :: tl(plev) ! Global grid temperature (K)
	 real(r8), intent(inout) :: ql(plev) ! Global grid water vapor (g/g)
	 real(r8), intent(inout) :: qcl(plev)! Global grid cloud liquid water (g/g)
	 real(r8), intent(inout) :: qci(plev)! Global grid cloud ice (g/g)
	 real(r8), intent(inout) :: ul(plev) ! Global grid u (m/s)
	 real(r8), intent(inout) :: vl(plev) ! Global grid v (m/s)
	 real(r8), intent(inout) :: u_crm  (crm_nx, crm_ny, crm_nz) ! CRM u-wind component
	 real(r8), intent(inout) :: v_crm  (crm_nx, crm_ny, crm_nz) ! CRM v-wind component
	 real(r8), intent(inout) :: w_crm  (crm_nx, crm_ny, crm_nz) ! CRM w-wind component
	 real(r8), intent(inout) :: t_crm  (crm_nx, crm_ny, crm_nz) ! CRM temperuture
	 real(r8), intent(inout) :: q_crm  (crm_nx, crm_ny, crm_nz) ! CRM total water
	 real(r8), intent(inout) :: qn_crm  (crm_nx, crm_ny, crm_nz) ! CRM cloud water/ice
	 real(r8), intent(inout) :: qp_crm (crm_nx, crm_ny, crm_nz)! CRM precipitation
         logical, intent(inout)  :: doinitial ! initialize run
	 real(r8), intent(inout) :: cltot ! shaded cloud fraction
	 real(r8), intent(inout) :: clhgh ! shaded cloud fraction
	 real(r8), intent(inout) :: clmed ! shaded cloud fraction
	 real(r8), intent(inout) :: cllow ! shaded cloud fraction
	 real(r8), intent(inout) :: stat_buffer(*)  ! buffer for one-column statistics
	 
!  Output

	 real(r8), intent(out) :: sltend(plev) ! tendency of static energy
	 real(r8), intent(out) :: qltend(plev) ! tendency of water vapor
	 real(r8), intent(out) :: qcltend(plev)! tendency of cloud liquid water
	 real(r8), intent(out) :: qcitend(plev)! tendency of cloud ice
	 real(r8), intent(out) :: ultend(plev) ! tendency of u
	 real(r8), intent(out) :: vltend(plev) ! tendency of v
	 real(r8), intent(out) :: t_rad (crm_nx, crm_ny, crm_nz) ! rad temperuture
	 real(r8), intent(out) :: qv_rad(crm_nx, crm_ny, crm_nz) ! rad vapor
	 real(r8), intent(out) :: qc_rad(crm_nx, crm_ny, crm_nz) ! rad cloud water
	 real(r8), intent(out) :: qi_rad(crm_nx, crm_ny, crm_nz) ! rad cloud ice
	 real(r8), intent(out) :: precc ! convective precip rate (m/s)
	 real(r8), intent(out) :: precl ! stratiform precip rate (m/s)
	 real(r8), intent(out) :: cld(plev)  ! cloud fraction
	 real(r8), intent(out) :: cldr(plev)  ! cloud fraction based on -30dBZ radar reflectivity
	 real(r8), intent(out) :: cldtop(plev)  ! cloud top pdf
	 real(r8), intent(out) :: gicewp(plev)  ! ice water path
	 real(r8), intent(out) :: gliqwp(plev)  ! ice water path
	 real(r8), intent(out) :: mc(plev)   ! cloud mass flux
	 real(r8), intent(out) :: mcup(plev) ! updraft cloud mass flux
	 real(r8), intent(out) :: mcdn(plev) ! downdraft cloud mass flux
	 real(r8), intent(out) :: mcuup(plev) ! unsat updraft cloud mass flux
	 real(r8), intent(out) :: mcudn(plev) ! unsat downdraft cloud mass flux
	 real(r8), intent(out) :: crm_qc(plev)  ! mean cloud water
	 real(r8), intent(out) :: crm_qi(plev)  ! mean cloud ice
	 real(r8), intent(out) :: crm_qs(plev)  ! mean snow
	 real(r8), intent(out) :: crm_qg(plev)  ! mean graupel
	 real(r8), intent(out) :: crm_qr(plev)  ! mean rain
	 real(r8), intent(out) :: flux_qt(plev) ! nonprecipitating water flux
	 real(r8), intent(out) :: flux_u(plev) ! 
	 real(r8), intent(out) :: flux_v(plev) ! 
         real(r8), intent(out) :: fluxsgs_qt(plev) ! sgs nonprecipitating water flux
         real(r8), intent(out) :: tkez(plev) ! tke profile
         real(r8), intent(out) :: tkesgsz(plev) ! sgs tke profile
	 real(r8), intent(out) :: flux_qp(plev) ! precipitating water flux
	 real(r8), intent(out) :: pflx(plev)    ! precipitation flux
	 real(r8), intent(out) :: qt_ls(plev) ! tendency of nonprec water due to large-scale
	 real(r8), intent(out) :: qt_trans(plev)! tendency of nonprec water due to transport
	 real(r8), intent(out) :: qp_trans(plev) ! tendency of prec water due to transport
	 real(r8), intent(out) :: qp_fall(plev) ! tendency of prec water due to fall-out
	 real(r8), intent(out) :: qp_src(plev) ! tendency of prec water due to conversion
	 real(r8), intent(out) :: qp_evp(plev) ! tendency of prec water due to evp
         real(r8), intent(out) :: t_ls(plev) ! tendency of lwse  due to large-scale
         real(r8), intent(out) :: u_ls(plev) 
         real(r8), intent(out) :: v_ls(plev) 
#ifdef CRMACCEL
         logical, intent(in) :: do_accel
         real(r8), intent(in) :: crm_accel_factor
#endif
         real(r8), intent(out) :: prectend ! column integrated tendency in precipitating water+ice (kg/m2/s)
         real(r8), intent(out) :: precstend ! column integrated tendency in precipitating ice (kg/m2/s)
	 real(r8), intent(out) :: precsc ! convective snow rate (m/s)
	 real(r8), intent(out) :: precsl ! stratiform snow rate (m/s)
         real(r8), intent(out):: taux_crm  ! zonal CRM surface stress perturbation (N/m2)
         real(r8), intent(out):: tauy_crm  ! merid CRM surface stress perturbation (N/m2)
         real(r8), intent(out):: z0m ! surface stress (N/m2)
         real(r8), intent(out):: timing_factor ! crm cpu efficiency
	 real(r8), intent(out)   :: qc_crm (crm_nx, crm_ny, crm_nz)! CRM cloud water
	 real(r8), intent(out)   :: qi_crm (crm_nx, crm_ny, crm_nz)! CRM cloud ice
	 real(r8), intent(out)   :: qpc_crm(crm_nx, crm_ny, crm_nz)! CRM precip water
	 real(r8), intent(out)   :: qpi_crm(crm_nx, crm_ny, crm_nz)! CRM precip ice
	 real(r8), intent(out)   :: prec_crm(crm_nx, crm_ny)! CRM precipiation rate
	 
	 
!  Local space:

	real zw(nz), dummy(nz), t00(nz)
	real tln(plev), qln(plev), qcln(plev), qcin(plev), uln(plev), vln(plev)
	real cwp(nx,ny), cwph(nx,ny), cwpm(nx,ny), cwpl(nx,ny)
	real factor_xy, idt_gl, omn, omp, omg, tmp1, tmp2
        real u2z,v2z,w2z
	integer i,j,k,l,ptop,nn,icyc, nstatsteps, ninterstop
	real, parameter :: umax = 0.5*crm_dx/crm_dt ! maxumum ampitude of the l.s. wind
	real, parameter :: wmin = 2.   ! minimum up/downdraft velocity for stat
        real, parameter :: cwp_threshold = 0.01 ! threshold for cloud condensate for shaded fraction calculation
	logical flag(12), flag_top(nx,ny), flagxy(nx,ny,4)
	real ustar, bflx, z0_est
	real colprec,colprecs
	real(r8) zs ! surface elevation
	real(r8) precold(nx,ny),prechist(nx,ny)
	integer nstepold
        logical stop_signal,dodiagnose
        character(6) text
!-----------------------------------------------
        dodiagnose = .false.
 1111   continue

!	idt_gl = 0.9999999/dt_gl
	idt_gl = 1._r8/dt_gl
	ptop = plev-nzm+1
	factor_xy = 1._r8/dble(nx*ny)
	dummy = 0.
	t_rad = 0.
	qv_rad = 0.
	qc_rad = 0.
	qi_rad = 0.
	zs=phis/gravit

	
! use climate model's constants

        cp = cpair
        lcond = latvap
        lfus = latice
        lsub = lcond+lfus
        ggr = gravit
	bflx = bflxls

        if(.not.doinitial) then
          ninterstop = nstop/nbreak
          goto 3333
        end if
 

!-----------------------------------------

        doinitial=.false.

	call setparm()

        if(ocnfrac.gt.0.5) then
           OCEAN = .true.
        else
           LAND = .true.
        end if




!        create CRM vertical grid and initialize some vertical reference arrays:
!
        do k = 1, nzm

           z(k) = zmid(plev-k+1) - zint(plev+1)
           zw(k) = zint(plev-k+2)- zint(plev+1)
           pres(k) = pmid(plev-k+1)/100.
           bet(k) = ggr/tl(plev-k+1)
           betp(k)= ggr/(pres(k)*100.)
           gamaz(k)=ggr/cp*z(k)

        end do ! k
	zw(nz) =  zint(plev-nz+2)

        dz = 0.5*(z(1)+z(2))
        do k=2,nzm
           adzw(k) = (z(k)-z(k-1))/dz
        end do
        adzw(1) = 1.
        adzw(nz) = adzw(nzm)
        adz(1) = 1.
        do k=2,nzm-1
          adz(k) = 0.5*(z(k+1)-z(k-1))/dz
        end do
        adz(nzm) = adzw(nzm)


        do k = 1,nzm
          rho(k) = pdel(plev-k+1)/ggr/(adz(k)*dz)
	end do
        do k=2,nzm
          rhow(k) = 0.5*(rho(k)+rho(k-1))
        end do
        rhow(1) = 2*rhow(2) - rhow(3)
        rhow(nz)= 2*rhow(nzm) - rhow(nzm-1)
	colprec=0
	colprecs=0

! limit the velocity at the very first step:

        if(u_crm(1,1,1).eq.u_crm(2,1,1).and.
     &                   u_crm(3,1,2).eq.u_crm(4,1,2)) then
         do k=1,nzm
          do j=1,ny
           do i=1,nx
             u_crm(i,j,k) = min( umax, max(-umax,u_crm(i,j,k)) )
#ifdef TESTCRMBUG
             v_crm(i,j,k) = 0.
#else
             v_crm(i,j,k) = min( umax, max(-umax,v_crm(i,j,k)) )
#endif
           end do
          end do
         end do

        end if

!
!  Initialize:
!
        do k=1,nzm

          u0(k)=0.
          v0(k)=0.
          t0(k)=0.
          t00(k)=0.
          tabs0(k)=0.
          q0(k)=0.
          qv0(k)=0.

          do j=1,ny
           do i=1,nx

            u(i,j,k)= u_crm(i,j,k)
#ifdef TESTCRMBUG
            v(i,j,k) =0.
#else
            v(i,j,k)= v_crm(i,j,k)
#endif
            w(i,j,k)= w_crm(i,j,k)
	    tabs(i,j,k) = t_crm(i,j,k)
	    q(i,j,k)= q_crm(i,j,k)
	    qn(i,j,k)= qn_crm(i,j,k)
            qp(i,j,k)= qp_crm(i,j,k)
	    tk(i,j,k) = 0.
	    tkh(i,j,k) = 0.
	    p(i,j,k) = 0.

	    do l=1,3
	     dudt(i,j,k,l) = 0.
	     dvdt(i,j,k,l) = 0.
	     dwdt(i,j,k,l) = 0.
	    end do

           omegan(i,j,k) =
     &       max(0.,min(1.,(tabs(i,j,k)-tbgmin)/(tbgmax-tbgmin)))
           omegap(i,j,k) =
     &       max(0.,min(1.,(tabs(i,j,k)-tprmin)/(tprmax-tprmin)))
           omegag(i,j,k) =
     &       max(0.,min(1.,(tabs(i,j,k)-tgrmin)/(tgrmax-tgrmin)))

	    omn = omegan(i,j,k)
	    omp = omegap(i,j,k) 

            t(i,j,k) = tabs(i,j,k)+gamaz(k)
     &       -(fac_cond+(1.-omn)*fac_fus)*qn(i,j,k)-
     &        (fac_cond+(1.-omp)*fac_fus)*qp(i,j,k)
	   
	    colprec=colprec+qp_crm(i,j,k)*pdel(plev-k+1)
	    colprecs=colprecs+qp_crm(i,j,k)*(1.-omp)*pdel(plev-k+1)
            u0(k)=u0(k)+u(i,j,k)
            v0(k)=v0(k)+v(i,j,k)
            t0(k)=t0(k)+t(i,j,k)
            t00(k)=t00(k)+t(i,j,k)+(fac_cond+(1.-omp)*fac_fus)*qp(i,j,k)
            tabs0(k)=tabs0(k)+tabs(i,j,k)
            q0(k)=q0(k)+q(i,j,k)
	    qv0(k) = qv0(k) + max(0.,q(i,j,k)-qn(i,j,k))

	   end do
	  end do

          u0(k) = u0(k) * factor_xy
          v0(k) = v0(k) * factor_xy
          t0(k) = t0(k) * factor_xy
          t00(k) = t00(k) * factor_xy
          tabs0(k) = tabs0(k) * factor_xy
          q0(k) = q0(k) * factor_xy
          qv0(k) = qv0(k) * factor_xy


	  l = plev-k+1
	  omg = (tl(l)-tbgmin)/(tbgmax-tbgmin)
	  omg = max(0.,min(1.,omg))
	  uln(l) = min( umax, max(-umax,ul(l)) )
#ifdef TESTCRMBUG
          vln(l) = 0.
#else
	  vln(l) = min( umax, max(-umax,vl(l)) )
#endif
	  ttend(k) = (tl(l)+gamaz(k)-
     &          fac_cond*(qcl(l)+qci(l))-fac_fus*qci(l)-t00(k))*idt_gl
!	  ttend(k) = (tl(l)-tabs0(k))*idt_gl
	  qtend(k) = (ql(l)+qcl(l)+qci(l)-q0(k))*idt_gl
	  utend(k) = (uln(l)-u0(k))*idt_gl
	  vtend(k) = (vln(l)-v0(k))*idt_gl
          ug0(k) = uln(l)
          vg0(k) = vln(l)
!	  tg0(k) = tl(l)+gamaz(k)-(fac_cond+(1.-omg)*fac_fus)*qcl(l)
	  tg0(k) = tl(l)+gamaz(k)-fac_cond*qcl(l)-fac_sub*qci(l)
	  qg0(k) = ql(l)+qcl(l)+qci(l)

	end do ! k

! save the 3D prognostic fields for diagnostic perposes in the event of a crash:

        u_save = u
        v_save = v
        w_save = w
        q_save = q
        qp_save = qp
        t_save = t


	uhl = u0(1)
	vhl = v0(1)

! estimate roughness length assuming logarithmic profile of velocity near the surface:  

	ustar = sqrt(tau00/rho(1))
        z0 = z0_est(z(1),bflx,wnd,ustar)
        z0 = max(0.00001,min(1.,z0))

        timing_factor = 0.

	prectend=colprec
	precstend=colprecs

        w(:,:,nz)=0.
        fluxbu=0.
        fluxbv=0.
        fluxbt=0.
        fluxbq=0.
        fluxtu=0.
        fluxtv=0.
        fluxtt=0.
        fluxtq=0.
        fzero =0.
        precsfc=0.
        precssfc=0.

!---------------------------------------------------	
	cld = 0.
	cldr = 0.
	cldtop = 0.
	gicewp=0
	gliqwp=0
	mc = 0.
	mcup = 0.
	mcdn = 0.
	mcuup = 0.
	mcudn = 0.
	crm_qc = 0.
	crm_qi = 0.
	crm_qs = 0.
	crm_qg = 0.
	crm_qr = 0.
        flux_qt = 0.
        flux_u = 0.
        flux_v = 0.
        fluxsgs_qt = 0.
        tkez = 0.
        tkesgsz = 0.
        flux_qp = 0.
        pflx = 0.
        qt_trans = 0.
        qp_trans = 0.
        qp_fall = 0.
        qp_evp = 0.
        qp_src = 0.
        qt_ls = 0.
        t_ls = 0.
        u_ls = 0.
        v_ls = 0.
	qwle = 0.
	qwsb = 0.
	qpwle = 0.
	qpwsb = 0.
	qadv = 0.
	qdiff = 0.
	qpadv = 0.
	qpdiff = 0.
	qpsrc = 0.
	qpfall = 0.
	qpevp = 0.
	precflux = 0.
!--------------------------------------------------
	call precip_init()

        if(u(1,1,1).eq.u(2,1,1).and.u(3,1,2).eq.u(4,1,2)) 
     &	            call setperturb()

	nstop = dt_gl/dt
#ifdef CRMACCEL
        if (do_accel) then
          nstop = dt_gl/dt/(crm_accel_factor+1)
          ! pritch, if accelerating by two, stop twice as soon
          
        end if
#endif
!	dt = dt_gl/nstop
        ! this feels redundant - unless it's a integer-conversion trick?
!	nsave3D = nint(60/dt)
!	if(nint(nsave3D*dt).ne.60)then
!	   print *,'CRM: time step=',dt,' is not divisible by 60 seconds'
!	   print *,'this is needed for output every 60 seconds'
!	   stop
!endif
	nstep = 0
	nprint = 1
	ncycle = 0
	day=0.
        ninterstop = nstop/nbreak
        if (mod(ninterstop,1).ne.0) then
           print*, 'CRM: ninterstop not integer.'
           stop
        end if

        if(mod(nstop,nbreak).ne.0) then
          print*, 'CRM: nstop is not divisible by nbreak:'
     &             ,nstop,nbreak
          stop
        end if

 3333   continue
        precold=precsfc
	nstepold=nstep

c------------------------------------------------------------------
c   Main time loop
c------------------------------------------------------------------

        do while(nstep.lt.nstop)

	  day = day + dt
#ifdef CRMACCEL
c INSERT pritch need to understand how this will work in the context of SP...
          day = day + crm_accel_factor*dt
#endif
          nstep = nstep + 1


c------------------------------------------------------------------
c  Check if the dynamical time step should be decreased
c  to handle the cases when the flow being locally linearly unstable
cc------------------------------------------------------------------

          ncycle = 1

          call kurant(stop_signal)

         if(stop_signal) then

           if(dodiagnose) call task_abort()

           write(text,'(i5 )') lchnk
           open(88,file='out'//trim(adjustl(text))//'.bin',form='unformatted')
           print*,'saved the diagnostics file ...'
           write(88) lchnk, icol, tl, ql, qcl, qci, ul, vl,
     &                  ps, pmid, pdel, phis, zmid, zint, dt_gl, plev,
     &                  qltend, qcltend, qcitend, sltend,
     &                  u_crm, v_crm, w_crm, t_crm, q_crm, qn_crm, qp_crm,
     &                  qc_crm, qi_crm, qpc_crm, qpi_crm, prec_crm,
     &                  qrs_crm, qrl_crm,
     &                  fsds, fsns, fsnt, fsut,
     &                  flwds, flns, flut, fsntc, fsdsc, flutc, flnsc,
     &                  t_rad, qv_rad, qc_rad, qi_rad,
     &                  precc, precl, precsc, precsl,
     &                  cltot, clhgh, clmed, cllow, 
     &                  cld, cldr, cldtop, gicewp, gliqwp, mc, mcup, mcdn, mcuup, mcudn,
     &                  crm_qc, crm_qi, crm_qs, crm_qg, crm_qr,
     &                  flux_qt, flux_qp, pflx, qt_ls,
     &                  qt_trans, qp_trans, qp_fall, qp_evp, qp_src, t_ls,
     &                  prectend, precstend,
     &                  nbreak, doinitial, ocnfrac,
     &                  wnd, tau00, bflxls, taux_crm, tauy_crm, z0m, timing_factor

            dodiagnose = .true.
            doinitial = .true.
            goto 1111


         end if

          do icyc=1,ncycle

	     icycle = icyc
             dtn = dt/ncycle
             dt3(na) = dtn
	     timing_factor = timing_factor+1

c---------------------------------------------
c       the Adams-Bashforth scheme in time

	   call t_startf ('crm_dycore')

             call abcoefs()

c---------------------------------------------
c       initialize stuff:

             call zero()

c-----------------------------------------------------------
c       Buoyancy term:

             call buoyancy()
c----------------------------------------------------------
c       suppress turbulence near the upper boundary (spange):

!#ifdef CRMACCEL
!        if(dodamping) call damping(do_accel,crm_accel_factor)
!#else
             if(dodamping) call damping()
!#endif
c------------------------------------------------------------
c       Large-scale and surface forcing:

             call forcing()

	     do k=1,nzm
	      do j=1,ny
	       do i=1,nx
	         t(i,j,k) = t(i,j,k) + (qrs_crm(i,j,k)+qrl_crm(i,j,k))*dtn
	       end do
	      end do
	     end do

c----------------------------------------------------------
c      Update the subdomain's boundaries for velocity

             call boundaries(0)

c---------------------------------------------------------
c       SGS TKE equation:

             if(dosgs) call tke_full()

c---------------------------------------------------------

c        Update boundaries for the SGS exchange coefficients:

          call boundaries(4)

c-----------------------------------------------
c       advection of momentum:

             call advect_mom()

c-----------------------------------------------
c       surface fluxes:

          if(dosurface) then

              call crmsurface(bflx)

          end if

c----------------------------------------------------------
c       SGS diffusion of momentum:

             if(dosgs) call diffuse_mom()

c---------------------------------------------------------
c       compute rhs of the Poisson equation and solve it for pressure.

             call pressure()

c--------------------------------------------------------
c       find velocity field at n+1/2 timestep needed for advection of scalars:


             call adams()

   	call t_stopf('crm_dycore')

c----------------------------------------------------------
c     Update boundaries for velocity fields to use for advection of
c     scalars:

             call boundaries(1)

c        Update boundaries for scalars:

             call boundaries(2)

c---------------------------------------------------------
c      advection of scalars :

  	 call t_startf ('crm_scalar')

        call advect_scalar(t,dummy,dummy)

        call advect_scalar(q,qadv,qwle)

        if(docloud.and.doprecip) then

          call advect_scalar(qp,qpadv,qpwle)

          call precip_fall()

        endif

c---------------------------------------------------------
c      diffusion of scalars :

          if(dosgs) then

c        Update boundaries for scalars:

            call boundaries(3)

            call diffuse_scalar(t,fluxbt,fluxtt,dummy,dummy)
            call diffuse_scalar(q,fluxbq,fluxtq,qdiff,qwsb)
            if(docloud.and.doprecip) then
              call diffuse_scalar(qp,fzero,fzero,qpdiff,qpwsb)
            endif

	   end if

	   call t_stopf('crm_scalar')

c-----------------------------------------------------------
c       Cloud condensation/evaporation and precipitation processes:


          if(docloud) then
	   call t_startf ('crm_cloud')
             call cloud()
	   call t_stopf('crm_cloud')
	   call t_startf ('crm_precip')
             if(doprecip) call precip_proc()
	   call t_stopf('crm_precip')
          end if

#ifdef CRMACCEL
          if (do_accel) then
! Use Jones-Bretherton methodology to accelerate
! CRM horizontal mean evolution artificially.
             call accelerate_crm(lchnk,icol,crm_accel_factor)
          endif
#endif

c-----------------------------------------------------------
c       Compute field diagnostics and update the velocity field:

             call diagnose()
	     
c----------------------------------------------------------
c Rotate the dynamic tendency arrays for Adams-bashforth scheme:
c
             nn=na
             na=nc
             nc=nb
             nb=nn

          end do ! icycle

c----------------------------------------------------------
        
!	  if(mod(nstep,nsave3D).eq.0)then
!	     prechist=(precsfc-precold)*dz/dt/dble(nstep-nstepold)/1000.
! MSP commenting this out. Suspect it is used to write CRM-scale data
! at a single grid point but that it also interferes with writing
! history files in continguous subdomains.

!	     call crmhistory(lchnk,icol,day, zs, prechist, qrs_crm,qrl_crm,
!     &	                fsds, fsns, fsnt, fsut,
!     &	                flwds, flns, flut, fsntc, flutc, fsdsc, flnsc, ps)
!             precold=precsfc
!	     nstepold=nstep
!          endif

        cwp = 0.
 	cwph = 0.
	cwpm = 0.
	cwpl = 0.

        flag(:) = .true.
        flag_top(:,:) = .true.
        flagxy(:,:,:) = .true.

	do k=1,nzm
	 l = plev-k+1
	 do j=1,ny
	  do i=1,nx

	   omn = omegan(i,j,k)
           omp = omegap(i,j,k)
           omg = omegag(i,j,k)

	   crm_qc(l) = crm_qc(l) + qn(i,j,k)*omn
	   crm_qi(l) = crm_qi(l) + qn(i,j,k)*(1.-omn)
	   crm_qr(l) = crm_qr(l) + qp(i,j,k)*omp
	   crm_qg(l) = crm_qg(l) + qp(i,j,k)*(1.-omp)*omg
	   crm_qs(l) = crm_qs(l) + qp(i,j,k)*(1.-omp)*(1.-omg)

	   
	   tmp1 = rho(nz-k)*adz(nz-k)*dz*qn(i,j,nz-k)
           cwp(i,j) = cwp(i,j)+tmp1
           if(cwp(i,j).gt.cwp_threshold.and.flag_top(i,j)) then
               cldtop(k) = cldtop(k) + 1
               flag_top(i,j) = .false.
           end if
           if(pres(nz-k).ge.700.) then
               cwpl(i,j) = cwpl(i,j)+tmp1
	   else if(pres(nz-k).lt.400.) then
               cwph(i,j) = cwph(i,j)+tmp1
	   else
               cwpm(i,j) = cwpm(i,j)+tmp1
	   end if

! one-column statistics:

           if(i.eq.1.and.j.eq.1) then
	    if(qn(i,j,k)*omn*rho(k).gt.0.136e-3.or. 
     &	       qn(i,j,k)*(1.-omn)*rho(k).gt.0.0165e-3.or.
     &         qp(i,j,k).gt.0.) then
	         if(flag(4)) stat_buffer(4) = stat_buffer(4) + 1.
                 flag(4) = .false.
                 if(pres(k).ge.700.) then
                   if(flag(1)) stat_buffer(1) = stat_buffer(1) + 1.
                   flag(1) = .false.
	         else if(pres(k).lt.400.) then
                   if(flag(3)) stat_buffer(3) = stat_buffer(3) + 1.
                   flag(3) = .false.
	         else
                   if(flag(2)) stat_buffer(2) = stat_buffer(2) + 1.
                   flag(2) = .false.
	         end if
            end if
            stat_buffer(5) = stat_buffer(5) + 
     &              (qn(i,j,k)*omn+qp(i,j,k)*omp)*rho(k)*adz(k)*dz
            stat_buffer(6) = stat_buffer(6) + 
     &         (qn(i,j,k)*(1.-omn)+qp(i,j,k)*(1.-omp))*rho(k)*adz(k)*dz
            if((qn(i,j,k)+qp(i,j,k))*rho(k).gt.5.e-8.and.flag(11)) then
              stat_buffer(7) = stat_buffer(7) + z(k)
              flag(11) = .false.
            end if
            if((qn(i,j,nz-k)+qp(i,j,nz-k))*rho(nz-k).gt.5.e-8
     &                                              .and.flag(12)) then
              stat_buffer(8) = stat_buffer(8) + z(k)
              flag(12) = .false.
            end if
           end if ! i.eq.1.and.j.eq.1

! Domain average cloud statistics based on radar reflectivity:

	    if(qn(i,j,k)*omn*rho(k).gt.0.136e-3.or. 
     &	       qn(i,j,k)*(1.-omn)*rho(k).gt.0.0165e-3.or.
     &         qp(i,j,k).gt.0.) then
	         if(flagxy(i,j,4)) stat_buffer(19) = stat_buffer(19) + 1.
                 flagxy(i,j,4) = .false.
                 if(pres(k).ge.700.) then
                   if(flagxy(i,j,1)) stat_buffer(16) = stat_buffer(16) + 1.
                   flagxy(i,j,1) = .false.
	         else if(pres(k).lt.400.) then
                   if(flagxy(i,j,3)) stat_buffer(18) = stat_buffer(18) + 1.
                   flagxy(i,j,3) = .false.
	         else
                   if(flagxy(i,j,2)) stat_buffer(17) = stat_buffer(17) + 1.
                   flagxy(i,j,2) = .false.
	         end if
            end if

	   if(qn(i,j,k)*omn*rho(k).gt.0.136e-3.or.
     &	       qn(i,j,k)*(1.-omn)*rho(k).gt.0.0165e-3
     &                                    .or.qp(i,j,k).gt.0.) then
               cldr(l) = cldr(l) + 1.
           end if

	   if(qn(i,j,k).gt.min(1.e-5,0.01*qsat(i,j,k))) then 
		cld(l) = cld(l) + 1.
	        if(w(i,j,k+1)+w(i,j,k).gt.2*wmin) then
		  mcup(l) = mcup(l) + rho(k)*0.5*(w(i,j,k+1)+w(i,j,k))
	        end if
	        if(w(i,j,k+1)+w(i,j,k).lt.-2*wmin) then
		  mcdn(l) = mcdn(l) + rho(k)*0.5*(w(i,j,k+1)+w(i,j,k))
	        end if
	   else 
	        if(w(i,j,k+1)+w(i,j,k).gt.2*wmin) then
		  mcuup(l) = mcuup(l) + rho(k)*0.5*(w(i,j,k+1)+w(i,j,k))
	        end if
	        if(w(i,j,k+1)+w(i,j,k).lt.-2*wmin) then
		  mcudn(l) = mcudn(l) + rho(k)*0.5*(w(i,j,k+1)+w(i,j,k))
	        end if
	   end if

	   t_rad (i,j,k) = t_rad (i,j,k)+tabs(i,j,k)
	   qv_rad(i,j,k) = qv_rad(i,j,k)+max(0.,q(i,j,k)-qn(i,j,k))
	   qc_rad(i,j,k) = qc_rad(i,j,k)+qn(i,j,k)*omn
	   qi_rad(i,j,k) = qi_rad(i,j,k)+qn(i,j,k)*(1.-omn)
	   gicewp(l)=gicewp(l)+qn(i,j,k)*(1.-omn)
	   gliqwp(l)=gliqwp(l)+qn(i,j,k)*omn

	  end do
	 end do
	end do

        do j=1,ny
         do i=1,nx
           if(cwp(i,j).gt.cwp_threshold) cltot = cltot + 1.
           if(cwph(i,j).gt.cwp_threshold) clhgh = clhgh + 1.
           if(cwpm(i,j).gt.cwp_threshold) clmed = clmed + 1.
           if(cwpl(i,j).gt.cwp_threshold) cllow = cllow + 1.
         end do
        end do

!
! return for cycling purposes:
!
        if(mod(nstep,ninterstop).eq.0) then
	  tmp1 = 1._r8/ dble(ninterstop)
	  t_rad = t_rad * tmp1
	  qv_rad = qv_rad * tmp1
	  qc_rad = qc_rad * tmp1
	  qi_rad = qi_rad * tmp1
          if (nstep.ne.nstop) return
        end if

         if(dodiagnose) call stepout()
c----------------------------------------------------------
        end do ! main loop
c----------------------------------------------------------


        tln(1:ptop-1) = tl(1:ptop-1)
        qln(1:ptop-1) = ql(1:ptop-1)
        qcln(1:ptop-1)= qcl(1:ptop-1)
        qcin(1:ptop-1)= qci(1:ptop-1)
        uln(1:ptop-1) = ul(1:ptop-1)
        vln(1:ptop-1) = vl(1:ptop-1)


!  Compute tendencies due to CRM:

	tln(ptop:plev) = 0.
	qln(ptop:plev) = 0.
	qcln(ptop:plev)= 0.
	qcin(ptop:plev)= 0.
	uln(ptop:plev) = 0.
	vln(ptop:plev) = 0.

	colprec=0
	colprecs=0
        do k = 1,nzm
	 l = plev-k+1
         do i=1,nx
          do j=1,ny
	     colprec=colprec+qp(i,j,k)*pdel(plev-k+1)
	     colprecs=colprecs+qp(i,j,k)*(1.-omegap(i,j,k))*pdel(plev-k+1)
             tln(l) = tln(l)+tabs(i,j,k)
             qln(l) = qln(l)+(q(i,j,k)-qn(i,j,k))
             qcln(l)= qcln(l)+qn(i,j,k)*omegan(i,j,k)
             qcin(l)= qcin(l)+qn(i,j,k)*(1.-omegan(i,j,k))
             uln(l) = uln(l)+u(i,j,k)
             vln(l) = vln(l)+v(i,j,k)
          end do ! k
         end do
        end do ! i
	

	tln(ptop:plev) = tln(ptop:plev) * factor_xy
	qln(ptop:plev) = qln(ptop:plev) * factor_xy
	qcln(ptop:plev) = qcln(ptop:plev) * factor_xy
	qcin(ptop:plev) = qcin(ptop:plev) * factor_xy
	uln(ptop:plev) = uln(ptop:plev) * factor_xy
	vln(ptop:plev) = vln(ptop:plev) * factor_xy
	
        sltend = cp * (tln - tl) * idt_gl
        ultend = (uln - ul) * idt_gl
        vltend = (vln - vl) * idt_gl
        qltend = (qln - ql) * idt_gl
        qcltend = (qcln - qcl) * idt_gl
        qcitend = (qcin - qci) * idt_gl
	prectend=(colprec-prectend)/ggr*factor_xy * idt_gl
	precstend=(colprecs-precstend)/ggr*factor_xy * idt_gl

	
!-------------------------------------------------------------
!
! Save the last step to the permanent core:

	u_crm  (1:nx,1:ny,1:nzm) = u   (1:nx,1:ny,1:nzm) 
	v_crm  (1:nx,1:ny,1:nzm) = v   (1:nx,1:ny,1:nzm) 
	w_crm  (1:nx,1:ny,1:nzm) = w   (1:nx,1:ny,1:nzm) 
	t_crm  (1:nx,1:ny,1:nzm) = tabs(1:nx,1:ny,1:nzm)
	q_crm  (1:nx,1:ny,1:nzm) = q   (1:nx,1:ny,1:nzm) 
	qn_crm (1:nx,1:ny,1:nzm) = qn  (1:nx,1:ny,1:nzm)
	qp_crm (1:nx,1:ny,1:nzm) = qp  (1:nx,1:ny,1:nzm)
	qc_crm (1:nx,1:ny,1:nzm)=qn(1:nx,1:ny,1:nzm)
     &                                    *omegan(1:nx,1:ny,1:nzm) 
	qi_crm (1:nx,1:ny,1:nzm)=qn(1:nx,1:ny,1:nzm) 
     &                        *(1.-omegan(1:nx,1:ny,1:nzm)) 
	qpc_crm(1:nx,1:ny,1:nzm)=qp(1:nx,1:ny,1:nzm)
     &                                    *omegap(1:nx,1:ny,1:nzm) 
	qpi_crm(1:nx,1:ny,1:nzm)=qp(1:nx,1:ny,1:nzm)
     &                        *(1.-omegap(1:nx,1:ny,1:nzm)) 

	z0m = z0
        taux_crm = taux0 / dble(nstop)
        tauy_crm = tauy0 / dble(nstop)
!---------------------------------------------------------------
!
!  Diagnostics:

	cld = min(1._r8,cld/float(nstop)*factor_xy)
	cldr = min(1._r8,cldr/float(nstop)*factor_xy) 
	cldtop = min(1._r8,cldtop/float(nstop)*factor_xy) 
	gicewp(:)=gicewp*pdel(:)*1000./gravit/float(nstop)*factor_xy
	gliqwp(:)=gliqwp*pdel(:)*1000./gravit/float(nstop)*factor_xy
	mcup = mcup / float(nstop) * factor_xy
	mcdn = mcdn / float(nstop) * factor_xy 
	mcuup = mcuup / float(nstop) * factor_xy 
	mcudn = mcudn / float(nstop) * factor_xy 
	mc = mcup + mcdn + mcuup + mcudn
	crm_qc = crm_qc / float(nstop) * factor_xy 
	crm_qi = crm_qi / float(nstop) * factor_xy 
	crm_qs = crm_qs / float(nstop) * factor_xy 
	crm_qg = crm_qg / float(nstop) * factor_xy
	crm_qr = crm_qr / float(nstop) * factor_xy

        precc = 0.
        precl = 0.
        precsc = 0.
        precsl = 0.
        do j=1,ny
         do i=1,nx
          precsfc(i,j) = precsfc(i,j)*dz/dt/dble(nstop)
          precssfc(i,j) = precssfc(i,j)*dz/dt/dble(nstop)
          if(precsfc(i,j).gt.10./86400.) then
             precc = precc + precsfc(i,j)
             precsc = precsc + precssfc(i,j)
          else
             precl = precl + precsfc(i,j)
             precsl = precsl + precssfc(i,j)
          end if
         end do
        end do
	prec_crm = precsfc/1000.
        precc = precc*factor_xy/1000.
        precl = precl*factor_xy/1000.
        precsc = precsc*factor_xy/1000.
        precsl = precsl*factor_xy/1000.

	cltot = cltot *factor_xy/nstop
	clhgh = clhgh *factor_xy/nstop
	clmed = clmed *factor_xy/nstop
	cllow = cllow *factor_xy/nstop

	stat_buffer(1:8) = stat_buffer(1:8) / dble(nstop)
	stat_buffer(16:19) = stat_buffer(16:19) * factor_xy / dble(nstop)

        stat_buffer(9) = precsfc(1,1)/1000.
c-------------------------------------------------------------
c       Fluxes and other stat:
c-------------------------------------------------------------
        do k=1,nzm
          u2z = 0.
          v2z = 0.
          w2z = 0.
          do j=1,ny
           do i=1,nx
             u2z = u2z+(u(i,j,k)-u0(k))**2
             v2z = v2z+(v(i,j,k)-v0(k))**2
             w2z = w2z+0.5*(w(i,j,k+1)**2+w(i,j,k)**2)
           end do
          end do
          tmp1 = dz/rhow(k)
          tmp2 = tmp1/dtn
          qwsb(k) = qwsb(k) * tmp1*rhow(k) * factor_xy/nstop
          qwle(k) = qwle(k) * tmp2*rhow(k) * factor_xy/nstop
          qpwsb(k) = qpwsb(k) * tmp1*rhow(k) * factor_xy/nstop
          qpwle(k) = qpwle(k) * tmp2*rhow(k) * factor_xy/nstop
	  qpadv(k) = qpadv(k) * factor_xy*idt_gl
	  qpdiff(k) = qpdiff(k) * factor_xy*idt_gl
	  qpsrc(k) = qpsrc(k) * factor_xy*idt_gl
	  qpevp(k) = qpevp(k) * factor_xy*idt_gl
	  qpfall(k) = qpfall(k) * factor_xy*idt_gl
	  qadv(k) = qadv(k) * factor_xy*idt_gl
	  qdiff(k) = qdiff(k) * factor_xy*idt_gl
	  precflux(k) = precflux(k) * factor_xy*dz/dt/nstop
	  l = plev-k+1
	  flux_qt(l) = qwle(k) + qwsb(k)
          fluxsgs_qt(l) =  qwsb(k)
          tkesgsz(l)= rho(k)*sum(tke(1:nx,1:ny,k))*factor_xy
#ifdef TESTCRMBUG
          tkez(l)= rho(k)*0.5*(u2z+v2z+w2z)*factor_xy + tkesgsz(l)
#else
          tkez(l)= rho(k)*0.5*(u2z+v2z*YES3D+w2z)*factor_xy + tkesgsz(l)
#endif
	  flux_qp(l) = qpwle(k) + qpwsb(k)
	  pflx(l) = precflux(k)/1000.
	  qt_trans(l) = qadv(k)+qdiff(k)
	  qp_trans(l) = qpadv(k)+qpdiff(k)
	  qp_fall(l) = qpfall(k)
	  qp_evp(l) = qpevp(k)
	  qp_src(l) = qpsrc(k)
	  qt_ls(l) = qtend(k)
	  t_ls(l) = ttend(k)
          u_ls(l) = utend(k)
          v_ls(l) = vtend(k)
	end do

	timing_factor = timing_factor / nstop

	return
	end

