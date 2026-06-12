      parameter(nbin=50,ncol=11,np=11,nmax=400000,nzbin = 9)
      real pos(3,nmax),vel(3,nmax),acc(3,nmax),pot(nmax),u(nmax)
      real rcyl(nmax),mass
      real sum(np,nbin),q(np,nmax),c(np,nbin)
      real dens(nbin,nzbin),vcirc(nbin,nzbin)
      character*30 asciifile

      pi = 3.1415927
      write(*,*) 'enter ascii file'
      read(*,*) asciifile
      open(file= asciifile,unit=10,status='old')
c      read(10,*) ngas,nhalo,ndisk
      read(10,*) ndisk
c      read(10,*) mass
c      do i=1,11+ngas+nhalo
c         read(10,*)
c      enddo
      rmax = 20.
      dr = rmax/float(nbin)
c
c     read in columns form ascii file
c
c     get the mass separately, units of 2.325e9 (GalactICS units, assumes G=1)
c
c     columns read into xxx are x,y,z,vx,vy,vz
c
      do i=1,ndisk
c         read(10,*) itmp,itmp,tmp,(pos(j,i),j=1,3),
c     +        (vel(j,i),j=1,3),(acc(j,i),j=1,3)
         read(10,*) mass,(pos(j,i),j=1,3),
     +        (vel(j,i),j=1,3)
      enddo
      close(10)
      vel = vel*100.
c
      do i=1,ndisk
         rcyl(i) = sqrt(pos(1,i)**2. + pos(2,i)**2.)
         q(1,i) = mass
         q(2,i) = pos(3,i)
         q(3,i) = q(2,i)**2.
         q(4,i) = vel(3,i)
         q(5,i) = q(4,i)**2.
         q(6,i) = (pos(1,i)*vel(2,i)-pos(2,i)*vel(1,i))/rcyl(i)
         q(7,i) = q(6,i)**2.
         q(8,i) = (pos(1,i)*vel(1,i)+pos(2,i)*vel(2,i))/rcyl(i)
         q(9,i) = q(8,i)**2.
         q(10,i) = pos(1,i)*acc(1,i) + pos(2,i)*acc(2,i)
         if(rcyl(i).gt.2.6.and.rcyl(i).lt.2.8) then
            write(19,*) (pos(j,i),j=1,3),(vel(j,i),j=1,3),
     +           (acc(j,i),j=1,3)
         endif
      enddo
c
c     we'll compute averages for np different quantities
c
      do i=1,np
         do j=1,nbin
            sum(i,j) = 0.
         enddo
      enddo
c

      write(*,*) 'total mass',ndisk*mass

      c = 0.
      do i=1,ndisk
         ir = int(rcyl(i)/dr)+1
         if(ir.gt.0.and.ir.le.nbin) then
            sum(1,ir) = sum(1,ir) + 1.
            do j=2,np
               y = q(j,i) - c(j,ir)
               t = sum(j,ir) + y
               c(j,ir) = (t - sum(j,ir)) - y
               sum(j,ir) = t
            enddo
         endif
      enddo

      open(file='profiles.out',unit=20,status='replace')

      do i=1,nbin
         write(*,*) sum(1,i)
         if(sum(1,i).gt.10.) then 
            r = (float(i)-0.5)*dr
            r1 = r-0.5*dr
            r2 = r+0.5*dr
            area = pi*(r2**2. - r1**2.)
            surfdens = sum(1,i)*mass/area
            do j=2,np
               sum(j,i) = sum(j,i)/sum(1,i)
            enddo
            sum(3,i) = sqrt(sum(3,i)-sum(2,i)**2.)
            sum(5,i) = sqrt(sum(5,i)-sum(4,i)**2.)
            sum(11,i) = sum(7,i)
            sum(7,i) = sqrt(sum(7,i)-sum(6,i)**2.)
            sum(9,i) = sqrt(sum(9,i)-sum(8,i)**2.)
            write(20,55) r,sum(2,i),sum(3,i),sum(4,i),sum(5,i),
     +           sum(6,i),sum(8,i),sum(9,i),surfdens,
     +           sum(10,i),sum(11,i)
         endif
      enddo
      close(20)
 55   format(16f14.5)

      dens = 0.
      vcirc = 0.

      do i=1,ndisk
         ir = int(rcyl(i)/dr)+1
         if(ir.gt.0.and.ir.le.nbin) then
            zmax = 2.*sum(3,ir)
            dz = 2.*zmax/float(nzbin-1)
            iz = nint((pos(3,i) + zmax)/dz) + 1
            if(iz.ge.1.and.iz.le.nzbin) then
               dens(ir,iz) = dens(ir,iz) + 1.
               vcirc(ir,iz) = vcirc(ir,iz) + q(6,i)
            endif
         endif
      enddo

      do ir = 1,nbin
         r = (float(ir)-0.5)*dr + rmin
         r1 = r-0.5*dr
         r2 = r+0.5*dr
         area = pi*(r2**2. - r1**2.)
         do iz = 1,nzbin
            if(dens(ir,iz).ge.1) then
               vcirc(ir,iz) = vcirc(ir,iz)/dens(ir,iz)
               dens(ir,iz) = log10(dens(ir,iz)/area)
            else
               vcirc(ir,iz) = 0
               dens(ir,iz) = -100.
            endif
         enddo
         write(30,*) r,(vcirc(ir,iz),iz=1,nzbin)
         write(40,*) r,(dens(ir,iz),iz=1,nzbin)
      enddo
      
      end
