      parameter(nbin=50,ncol=11,np=9,nmax=100000)
      real pos(3,nmax),vel(3,nmax),acc(3,nmax),pot(nmax),u(nmax)
      real rcyl(nmax),mass
      real sum(np,nbin),q(np,nmax)
      character*30 asciifile

      pi = 3.1415927
      write(*,*) 'enter ascii file'
      read(*,*) asciifile
      open(file= asciifile,unit=10,status='old')
      read(10,*) ngas
      rmax = 20.
      rmin = 0.
      dr = (rmax-rmin)/float(nbin)
c
c     read in columns form ascii file
c
c     get the mass separately, units of 2.325e9 (GalactICS units, assumes G=1)
c
c     columns read into xxx are x,y,z,vx,vy,vz
c
      do i=1,ngas
         read(10,*) mass,(pos(j,i),j=1,3),
     +        (vel(j,i),j=1,3)
      enddo
      close(10)
c
      do i=1,ngas
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
      do i=1,ngas
         ir = int((rcyl(i)-rmin)/dr)+1
         if(ir.gt.0.and.ir.le.nbin) then
            sum(1,ir) = sum(1,ir) + 1.
            do j=2,np
               sum(j,ir) = sum(j,ir) + q(j,i)
            enddo
         endif
      enddo

      open(file='profiles.out',unit=20,status='replace')

      do i=1,nbin
         write(*,*) sum(1,i)
         if(sum(1,i).gt.10.) then 
            r = (float(i)-0.5)*dr + rmin
            r1 = r-0.5*dr
            r2 = r+0.5*dr
            area = pi*(r2**2. - r1**2.)
            do j=2,np
               sum(j,i) = sum(j,i)/sum(1,i)
            enddo
            sum(3,i) = sqrt(sum(3,i)-sum(2,i)**2.)
            sum(5,i) = sqrt(sum(5,i)-sum(4,i)**2.)
            sum(7,i) = sqrt(sum(7,i)-sum(6,i)**2.)
            sum(9,i) = sqrt(sum(9,i)-sum(8,i)**2.)
            write(20,55) r,log10(sum(1,i)/area),
     +           sum(2,i),sum(3,i),sum(4,i),sum(5,i),
     +           sum(6,i),sum(7,i),sum(8,i),sum(9,i)
         endif
      enddo
      close(20)
 55   format(16f14.5)
      end
