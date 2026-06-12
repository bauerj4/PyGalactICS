      include 'commonblocks'
      include 'equivalence'
      real psi,modelmag,kpctoarcsec
      real r(1000),vt(1000),vh(1000),vb(1000),v_obs(1000)
      character*30 filename


      ndata = 1000
      kpctoarcsec = 3600.*360/2./pi/distance
      kpctoarcmin = 60.*360/2./pi/distance
      chisqr = 0.

      open(40,file='vc.out',status='replace')
      filename = 'dbh.dat'
      call readharmfile(filename,ibuf,jbuf,kbuf)
      ibf = ibulgeflag

      nd = 300
      rmax = 100.
      drr = rmax/float(nd)

      do i=1,nd
         r(i) = float(i)*drr
         call force(r(i),0.,vt(i),fz,psi)
         vt(i) = sqrt(-r(i)*vt(i))
      enddo

      filename = 'h.dat'
      call readharmfile(filename,ibuf,jbuf,kbuf)

      do i=1,nd
         call force(r(i),0.,vh(i),fz,psi)
         vh(i) = sqrt(-r(i)*vh(i))
      enddo
      
      diskmass = 100/2.325
      c = 6.76

      if(ibf.eq.1) then
         filename = 'b.dat'
         call readharmfile(filename,ibuf,jbuf,kbuf)
         do i=1,nd
            call force(r(i),0.,vb(i),fz,psi)
            vb(i) = sqrt(-r(i)*vb(i))
            vd  = sqrt(max(0.,vt(i)**2. - vh(i)**2. - vb(i)**2.))
             write(40,*) r(i),vd,vb(i),vh(i),vt(i)
         enddo
      else
         do i=1,nd
            vd  = sqrt(vt(i)**2. - vh(i)**2.)
            write(40,*) r(i),vd,vh(i),vt(i)
         enddo
      endif

      close(40)
      return
      end

      FUNCTION PLGNDR1(L,X)
      PMM=1.
      IF(L.EQ.0) THEN
        PLGNDR1=PMM
      ELSE
        PMMP1=X*PMM
        IF(L.EQ.1) THEN
          PLGNDR1=PMMP1
        ELSE
          DO 12 LL=2,L
            PLL=(X*(2*LL-1)*PMMP1-(LL-1)*PMM)/(LL)
            PMM=PMMP1
            PMMP1=PLL
12        CONTINUE
          PLGNDR1=PLL
        ENDIF
      ENDIF
      RETURN
      END

c     this file contains the following subroutines:
c
c     readharmfile: reads a file with the harmonic expansion of the
c     potential and density.  all the relevant stuff ends up in a common
c     block that only needs to be seen by the routines in this file (I
c     think!). This routine must be called first.
c
c     dens(r,z): density at point (r,z)
c
c     densrpsi(r,psi): density as a function of r and potential psi.
c
c     pot(r,z): potential at point (r,z)
c
c     all the rest is (slightly nobbled in places) numerical recipes
c     routines.
c
      subroutine readharmfile(filename,ibuf1,jbuf1,kbuf1)

      include 'commonblocks'
      include 'equivalence'

      character*30 filename

      real psi0mpsid

      open(file='in.gendenspsi',unit=1,status='old')
      read(1,*) npsi
      close(1)

c  constants for legendre polynomials
      do l=0,40
         plcon(l) = sqrt((2*l+1)/(4.0*pi))
      enddo
c
      open(14,file=filename,status='old')
      read(14,*)
      read(14,'(2x,7g15.5,i6,i4)') chalo,v0,a,nnn,v0bulge,abulge,
     +     dr,nr,lmax
      read(14,*)
      read(14,'(2x,3g15.8)') psi0, haloconst, bulgeconst
      read(14,*)
      read(14,'(2x,5g15.5)') rmdisk, rdisk, zdisk, outdisk, drtrunc
      read(14,*)
      read(14,'(2x,5g15.5)') rmdisk2, rdisk2, zdisk2, outdisk2, drtrunc2
      read(14,*)
      read(14,'(2x,8g15.5)') rmgas,rgas,outgas,zgas0,drtruncgas,rzgas,
     +     zgasmax,gamma
      read(14,*)
      read(14,'(2x,3g15.5)') psic,psi0mpsid,bhmass
      psid = psi0 - psi0mpsid
      read(14,'(2x,6i5)') idiskflag, idiskflag2,igasflag,
     +     ibulgeflag, ihaloflag, ibhflag
      read(14,*)
      do ir=0,nr
         read(14,'(8g16.8)') rdummy,(adens(l/2+1,ir),l=0,lmax,2)
         enddo
      read(14,*)
      read(14,*)
      do ir=0,nr
         read(14,'(8g16.8)') rdummy,(apot(l/2+1,ir),l=0,lmax,2)
         enddo
      read(14,*)
      read(14,*)
      do ir=0,nr
         read(14,'(8g16.8)') rdummy,(fr(l/2+1,ir),l=0,lmax,2)
         enddo
      close(14)

c
c these are needed for the bulgeconstants common block
c
      sigbulge2=sigbulge*sigbulge
c
c additional disk constants for efficiency - common diskconstants
c

      nrdisk= int((outdisk + 2.0*drtrunc)/dr) + 10
      diskconst = rmdisk/(4.0*pi*rdisk*rdisk*zdisk)

      nrdisk2 = int((outdisk + 2.0*drtrunc2)/dr) + 10
      diskconst2 = rmdisk2/(4.0*pi*rdisk2*rdisk2*zdisk2)

      gasconst = rmgas/(2.0*pi*rgas*rgas)
c
c and some more constants...
c
      v02=v0*v0
      v03=v0*v02
c
c calculate potential correction
c
      redge = nr*dr
      potcor = 0
      do l=0,lmax,2
          potcor = potcor + apot(l/2+1,nr) + fr(l/2+1,nr)*redge/(l+1)
      enddo
      potcor = potcor*plcon(0)
      potcor1 = potcor
c
c transfer gparams buffer for output
c

      do i=1,20
         ibuf1(i) = ibuf(i)
      enddo
      do j=1,3
         jbuf1(j) = jbuf(j)
      enddo
      do k=1,1
         kbuf1(k) = kbuf(k)
      enddo
      
      return
      end

      function appdiskdens(s,z)

      include 'commonblocks'

c  This is the density corresponding to the first-guess disk potential
c  f(r)*erfc((r-outdisk)/sqrt(2)drtrunc)/2 * 4 pi G zdisk**2 log(z/zdisk)
c  where r is spherical radius. f(r) is here taken as an exponential.
c
c  The corresponding density is:
c  f(r)*erfc*sech(z/zdisk)**2 + radial gradient terms.
c  For radii below one scale radius, we have replaced the radial exponential 
c  (which gives rise to a singular laplacian) with a quartic that joins on 
c  smoothly.
c
      r=sqrt(s*s+z*z)
c     radial truncation factors
      t=sqrt(0.5d0)*(r-outdisk)/drtrunc
      t2=t*t
      if (t.lt.-4.0) then
         eexp=0.
         eerfc=1.
      elseif (t.lt.4.0) then
         eexp=exp(-t2)/sqrt(2*pi)/drtrunc
         eerfc=0.5*erfc(t)
      else
         eexp=0
         eerfc=0
      endif
c     radial density 
c     f is radial density, f1r is f'/r, f2 is f".
      if (r.gt.0.) then
         fac1=diskconst*zdisk**2*exp(-r/rdisk)
         f=fac1*eerfc
         f1r=-fac1*(eerfc/rdisk+eexp)/r
         f2=fac1*(eerfc/rdisk/rdisk
     +        +eexp*(2/rdisk+(r-outdisk)/drtrunc**2))
      else
         fac1=diskconst*zdisk**2
         f=fac1*eerfc
         f1r=0
         f2=0
      endif
c     vertical factors
      zz=abs(z/zdisk)
      ezz=exp(-zz)
      e2zz=ezz*ezz
      tlncosh=zz+log(0.5*(1+e2zz))
      tztanh=zz*(1-e2zz)/(1+e2zz)
      tsech2=(2*ezz/(1+e2zz))**2
      appdiskdens=f2*tlncosh+2*f1r*(tztanh+tlncosh)+f*tsech2/zdisk**2

      return
      end

      function appdiskdens2(s,z)

      include 'commonblocks'

      r=sqrt(s*s+z*z)
      t=sqrt(0.5d0)*(r-outdisk2)/drtrunc2
      t2=t*t
      if (t.lt.-4.0) then
         eexp=0.
         eerfc=1.
      elseif (t.lt.4.0) then
         eexp=exp(-t2)/sqrt(2*pi)/drtrunc2
         eerfc=0.5*erfc(t)
      else
         eexp=0
         eerfc=0
      endif

      if (r.gt.0.) then
         fac1=diskconst2*zdisk2**2*exp(-r/rdisk2)
         f=fac1*eerfc
         f1r=-fac1*(eerfc/rdisk2+eexp)/r
         f2=fac1*(eerfc/rdisk2/rdisk2+
     +        eexp*(2/rdisk2+(r-outdisk2)/drtrunc2**2))
      else
         fac1=diskconst2*zdisk2**2
         f=fac1*eerfc
         f1r=0
         f2=0
      endif

      zz=abs(z/zdisk2)
      ezz=exp(-zz)
      e2zz=ezz*ezz
      tlncosh=zz+log(0.5*(1+e2zz))
      tztanh=zz*(1-e2zz)/(1+e2zz)
      tsech2=(2*ezz/(1+e2zz))**2
      appdiskdens2=f2*tlncosh+2*f1r*(tztanh+tlncosh)+f*tsech2/zdisk2**2

      return
      end

      function appgasdens(s,z)

      include 'commonblocks'

      r=sqrt(s*s+z*z)

      call gassurfdens(r,f,f1r,f2)
      call gasscaleheight(s,h,h1,h2)
      call gasvertdens(z,h,g,g1,g2)

      if(s.gt.0.) then
         appgasdens = 
     +        f2*h*g +
     +        2*f1r*g*(h + h1*s) +
     +        2.*f1r*g1*z*(1.-s*h1/h) + 
     +        f*g2/h*(1. + (z*h1/h)**2.) -
     +        f*g1*z/h*(h2+ h1/s) +
     +        f*g*(h2 + h1/s)
      else
         appgasdens = 
     +        f2*h*g +
     +        2*f1r*g*h +
     +        2.*f1r*g1*z*(1.-s*h1/h) + 
     +        f*g2/h*(1. + (z*h1/h)**2.) -
     +        f*g1*z/h*h2 +
     +        f*g*h2
      endif


      if(isnan(appgasdens)) then
         write(0,*) 'nan for appgasdens r,z,f,f1r,f2',
     +        r,z,f,f1r,f2
         write(0,*) h,h1,h2
         write(0,*) g,g1,g2
         stop
      endif

      return
      end

      subroutine gasscaleheight(r,h,h1,h2)

      h = getzgas(r)
      h1 = getdzgas(r)
      h2 = getd2zgas(r)

      return
      end

      subroutine gasvertdens(z,zgas,g,g1,g2)
      include 'commonblocks'

      zz = z/zgas
      if(abs(zz).lt.1.) then
         g = 3./8.*(zz**2. - (zz**4.)/6.)
         g1 = 3./4.*(zz - zz**3./3.)
         g2 = 3./4.*(1.-zz**2.)
      else
         g = 0.5*(abs(zz)-3./8.)
         g1 = 0.5*sgn(zz)
         g2 = 0.
      endif

      return
      end

      subroutine gassurfdens(r,f,f1r,f2)

      include 'commonblocks'

      zgas = getzgas(r)

      t=sqrt(0.5d0)*(r-outgas)/drtruncgas
      t2=t*t
      if (t.lt.-4.0) then
         eexp=0.
         eerfc=1.
      elseif (t.lt.4.0) then
         eexp=exp(-t2)/sqrt(2*pi)/drtruncgas
         eerfc=0.5*erfc(t)
      else
         eexp=0
         eerfc=0
      endif

      if (r.gt.0.) then
         fac1 = gasconst*exp(-r/rgas)
         f = fac1*eerfc
         f1r = -fac1*(eerfc/rgas+eexp)/r
         f2 = fac1*(eerfc/rgas/rgas+
     +        eexp*(2/rgas+(r-outgas)/drtruncgas**2))
      else
         fac1 = gasconst
         f = fac1*eerfc
         f1r = 0
         f2 = 0
      endif

      if(isnan(f).or.f.gt.1e10) then
         write(*,*) 'inside gassurf fac1,eerfc,r,rgas,gasconst,zgas',
     +        fac1,eerfc,r,rgas,gasconst,zgas
         stop
      endif

      return
      end

      function getzgas(r)
      
      include 'commonblocks'

      if(rzgas.lt.0.) then
         getzgas = zgas0
      else
         ezgas = exp(-r/rzgas)
         getzgas = (zgasmax+1.)*zgas0/(zgasmax*ezgas + 1.)
      endif

      return
      end

      function getdzgas(r)
      
      include 'commonblocks'

      if(rzgas.lt.0.) then
         getdzgas = 0.
      else
         ezgas = exp(-r/rzgas)
         rnum = (zgasmax+1.)*zgas0*ezgas*zgasmax
         den = rzgas*((zgasmax*ezgas + 1.)**2.)
         getdzgas = rnum/den
      endif

      return
      end

      function getd2zgas(r)
      
      include 'commonblocks'

      if(rzgas.lt.0.) then
         getd2zgas = 0.
      else
         ezgas = exp(-r/rzgas)
         rnum = (zgasmax+1.)*zgas0*ezgas*(1. - ezgas*zgasmax)
         den = rzgas*rzgas*((ezgas*zgasmax + 1.)**3.)
         getd2zgas = rnum/den
      endif

      return
      end

      function appdiskpot(s,z)

      include 'commonblocks'

      r=sqrt(s*s+z*z)

      t=sqrt(0.5d0)*(r-outdisk)/drtrunc

      if (t.lt.-4.0) then
         eerfc=1.
      elseif(t.gt.4.0) then
         appdiskpot=0.
         return
      else
         eerfc=0.5*erfc(t)
      endif

      f=diskconst*exp(-r/rdisk)
      zz=abs(z/zdisk)
      appdiskpot=-4*pi*f*zdisk**2*(zz+log(0.5*(1+exp(-2*zz))))*eerfc

      return
      end       

      function appdiskpot2(s,z)

      include 'commonblocks'

      r=sqrt(s*s+z*z)

      t=sqrt(0.5d0)*(r-outdisk2)/drtrunc2
      if (t.lt.-4.0) then
         eerfc=1.
      elseif(t.gt.4.0) then
         appdiskpot2=0.
         return
      else
         eerfc=0.5*erfc(t)
      endif

      f=diskconst2*exp(-r/rdisk2)
      zz=abs(z/zdisk2)
      appdiskpot2=-4*pi*f*zdisk2**2*(zz+log(0.5*(1+exp(-2*zz))))*eerfc

      return
      end       

      function appgaspot(s,z)

      include 'commonblocks'

      r=sqrt(s*s+z*z)
      zgas = getzgas(s)
      
      call gassurfdens(r,f,f1r,f2)
      call gasvertdens(z,zgas,g,g1,g2)

      appgaspot = -4*pi*f*zgas*g

      if(isnan(appgaspot)) then
         write(*,*) 'inside appgaspot s,z,f,g,zgas',s,z,f,g,zgas
         stop
      endif

      return
      end       
      subroutine appdiskforce(s,z,fsad,fzad)

      include 'commonblocks'

      r=sqrt(s*s + z*z)
      t=sqrt(0.5d0)*(r-outdisk)/drtrunc
      if (t.lt.-4.0) then
          eerfc=1.
          derfc=0.0
      elseif(t.gt.4.0) then
          eerfc = 0.0
          derfc = 0.0
      else
          eerfc=0.5*erfc(t)
          derfc = exp(-t**2)/(sqrt(2.0*pi)*drtrunc)
      endif

      r1 = r/rdisk
      if (r1.eq.0.) then
         fprime=0
         f=0
      else
          texp = diskconst*zdisk**2*exp(-r1)
          f = texp*eerfc
          fprime = -texp*(derfc + eerfc/rdisk)/r
      endif

      zz = abs(z/zdisk)
      if( zz .lt. 10 ) then
          e2zz = exp(-2.0*zz)
      else
          e2zz = 0.0
      endif
      tlncoshz = zz + log(0.5*(1.0 + e2zz))

      fsad = -4.0*pi*fprime*s*tlncoshz
      fzad = -4.0*pi*(fprime*z*tlncoshz + f/zdisk*tanh(z/zdisk))

      return
      end

      subroutine appdiskforce2(s,z,fsad,fzad)

      include 'commonblocks'

      r=sqrt(s*s + z*z)
      t=sqrt(0.5d0)*(r-outdisk2)/drtrunc2
      if (t.lt.-4.0) then
          eerfc=1.
          derfc=0.0
      elseif(t.gt.4.0) then
          eerfc = 0.0
          derfc = 0.0
      else
          eerfc=0.5*erfc(t)
          derfc = exp(-t**2)/(sqrt(2.0*pi)*drtrunc2)
      endif

      r1 = r/rdisk2
      if (r1.eq.0.) then
         fprime=0
         f=0
      else
          texp = diskconst2*zdisk2**2*exp(-r1)
          f = texp*eerfc
          fprime = -texp*(derfc + eerfc/rdisk2)/r
      endif

      zz = abs(z/zdisk2)
      if( zz .lt. 10 ) then
          e2zz = exp(-2.0*zz)
      else
          e2zz = 0.0
      endif
      tlncoshz = zz + log(0.5*(1.0 + e2zz))

      fsad = -4.0*pi*fprime*s*tlncoshz
      fzad = -4.0*pi*(fprime*z*tlncoshz + f/zdisk2*tanh(z/zdisk2))

      return
      end


      subroutine appgasforce(s,z,fsad,fzad)

      include 'commonblocks'

      r=sqrt(s*s + z*z)

      call gassurfdens(r,f,f1r,f2)
      call gasscaleheight(s,h,h1,h2)
      call gasvertdens(z,h,g,g1,g2)

      fsad = -4.0*pi*(s*f1r*h*g + f*h1*g - z*h1/h*f*g1)
      fzad = -4.0*pi*(z*f1r*h*g + f*g1)

      return
      end

      function sgn(z)

      if(z.gt.0.) then
         sgn = 1.
      else
         if(z.lt.0.) then
            sgn = -1.
         else
            sgn = 0.
         endif
      endif
      return
      end
      subroutine force(s,z,fs,fz,pot)

      parameter (pc0=0.282094792, pc2=0.630783131, pc4=0.846284375)
      parameter (pc6=1.017107236, pc8=1.163106623)
      real pc(20), p(20), dp(20)
      real pot

      include 'commonblocks'

      r=sqrt(s*s+z*z)
      ihi=int(r/dr)+1
      if (ihi.lt.1) ihi=1
      if (ihi.gt.nr) ihi=nr
      r1=dr*(ihi-1)
      r2=dr*ihi
      redge = nr*dr
      t=(r-r1)/(r2-r1)
      ihim1 = ihi - 1
      tm1 = 1.0 - t
      if (r.eq.0.) then
         fs = 0.0
         fz = 0.0
      else
         costheta=z/r
         ct2 = costheta*costheta
         sintheta=s/r
         
         do l=0,lmax,2
            pc(l/2+1) = sqrt((2.0*l + 1)/(4.0*pi))
            p(l/2+1) = plgndr1(l,costheta) 
            if( costheta .eq. 1.0 ) then
               dp(l/2+1) = 0.0
            else
               st2 = 1.0 - costheta*costheta
               dp(l/2+1) = l*(plgndr1(l-1, costheta) - 
     +              costheta*p(l/2+1))/st2
            endif
         enddo
         do i=1,lmax/2+1
            p(i) = p(i)*pc(i)
            dp(i) = dp(i)*pc(i)
         enddo

         if( r .le. redge ) then
             frr = 0.0
             do i=1,lmax/2+1
                frr = frr + p(i)*(t*fr(i,ihi) + tm1*fr(i,ihim1))
             enddo
             fth = 0.0
             do i=2,lmax/2+1
                fth = fth - sintheta*dp(i)*(t*apot(i,ihi) + 
     +               tm1*apot(i,ihim1))
             enddo
             pot = 0.0
             do i=1,lmax/2+1
                pot = pot + p(i)*(t*apot(i,ihi) + tm1*apot(i,ihim1))
             enddo
         else
             frr = 0.0
             do i=1,lmax/2+1
                 l = 2*(i-1)
                 frr = frr-(l+1)*p(i)*apot(i,nr)/redge*(redge/r)**(l+2)
             enddo
             fth = 0.0
             do i=2,lmax/2+1
                 l = 2*(i-1)
                 fth = fth - sintheta*dp(i)*apot(i,nr)*(redge/r)**(l+1)
             enddo
             pot = 0.0
             do i=1,lmax/2+1
                l = 2*(i-1)
                pot = pot + p(i)*apot(i,nr)*(redge/r)**(l+1)
             enddo
         endif
         if( idiskflag .eq. 1 ) then
             pot = pot + appdiskpot(s,z)
         endif

         if( idiskflag2 .eq. 1 ) then
            pot = pot + appdiskpot2(s,z)
         endif

         if( igasflag .eq. 1 ) then
            pot = pot + appgaspot(s,z)
         endif

         fth = -fth

         fs = -(sintheta*frr + costheta/r*fth)
         fz = -(costheta*frr - sintheta/r*fth)

         if( idiskflag .eq. 1 ) then
            call appdiskforce(s,z,fsad,fzad)
            fs = fs + fsad
            fz = fz + fzad
         endif

         if( idiskflag2 .eq. 1 ) then
            call appdiskforce2(s,z,fsad,fzad)
            fs = fs + fsad
            fz = fz + fzad
         endif

         if( igasflag .eq. 1 ) then
            call appgasforce(s,z,fsad,fzad)
            fs = fs + fsad
            fz = fz + fzad
         endif

      endif
      return
      end

