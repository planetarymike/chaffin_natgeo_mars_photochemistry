##run_and_plot.jl --- routines to run the coupled photochemistry model
##to produce model output useful for a 2016 photochemistry paper

@everywhere @time update!(n_current,0.)

#we can run with logarithmic time steps
@everywhere timepts=logspace(log10(1),log10(1e7*3.14e7),1000)
@everywhere timediff=timepts[2:end]-timepts[1:end-1]
@everywhere append!(timediff,3.3e6*3.14e7*ones(Float64,300))


waterppmvec=[20 40 60 80]
wateraltvec = [20 40 60 80 100 120]
parmsvec=[[a,b] for a in waterppmvec, b in wateraltvec]
parmsvec=reshape(parmsvec,length(parmsvec))
filenamevec=[string("./ppm_",a[1],"_alt_",a[2],".h5") for a in parmsvec]

@everywhere function runwaterprofile(n_current,ppmadd,peakalt,dtlist,filename)
    n_internal=deepcopy(n_current)
    
    waterppm=1e-6*map(x->ppmadd.*exp(-((x-peakalt)/12.5)^2),alt[2:end-1]/1e5)+H2Oinitfrac
    waterprofile=waterppm.*map(z->n_tot(n_internal,z),alt[2:end-1])
    ##modify the water profile stored in n_internal
    n_internal[:H2O]=waterprofile

    runprofile(n_internal, dtlist, filename)

    return()
end


@everywhere function runprofile(n_current, dtlist, filename)
    n_internal=deepcopy(n_current)
    
    elapsed_time=0.0

    n_internal_mat=Array(Float64,length(alt)-2,length(collect(keys(n_internal))));
    for ispecies in 1:length(collect(keys(n_internal)))
        for ialt in 1:length(alt)-2
            n_internal_mat[ialt,ispecies]=n_internal[collect(keys(n_internal))[ispecies]][ialt]
        end
    end

    h5write(filename,"n_current/init",n_internal_mat)
    h5write(filename,"n_current/alt",alt)
    h5write(filename,"n_current/species",map(string,collect(keys(n_internal))))
    h5write(filename,"n_current/timelist",cumsum(dtlist))

    thisi=0
    for dt in dtlist
        println(filename*": iteration = "* string(thisi+=1)*" "*Libc.strftime(time()))
        println("dt = "* string(dt::Float64))
        elapsed_time+=dt
        println("elapsed_time = "*string(elapsed_time))
        ##n_old=deepcopy(n_internal)
        update!(n_internal,dt)

        ## save the concentrations to history
        ## write n_current into n_current_mat
        n_internal_mat=Array(Float64,length(alt)-2,length(collect(keys(n_internal))));
        for ispecies in 1:length(collect(keys(n_internal)))
            for ialt in 1:length(alt)-2
                n_internal_mat[ialt,ispecies]=n_internal[collect(keys(n_internal))[ispecies]][ialt]
            end
        end
        ## write n_internal_mat to file
        h5write(filename,
                string("n_current/iter_",thisi),
                n_internal_mat)
    end
    return n_internal
end

@everywhere function read_ncurrent_from_file(readfile,tag)
    thisalt=h5read(readfile,"n_current/alt")
    if thisalt!=alt
        throw("altitudes in file do not match altitudes in memory!")
    end
    n_current_tag_list=map(symbol,h5read(readfile,"n_current/species"))
    n_current_mat=h5read(readfile,tag);
    n_current=Dict{Symbol,Array{Float64,1}}()
    for ispecies in [1:length(n_current_tag_list);]
        n_current[n_current_tag_list[ispecies]]=reshape(n_current_mat[:,ispecies],length(alt)-2)
    end
    n_current
end

@everywhere function get_H_fluxes(readfile)
    mydset=h5open(readfile,"r")
    mydata=read(mydset)
    timelength=length(mydata["n_current"]["timelist"])+1
    close(mydset)
    Hfluxes=fill(0.,timelength)
    n_current=read_ncurrent_from_file(readfile,string("n_current/init"))
    Hfluxes[1]=(n_current[:H][end]*speciesbcs(:H)[2,2]
                  +2*n_current[:H2][end]*speciesbcs(:H2)[2,2])
    for i in 1:(timelength-1)
        #println(i)
        n_current=read_ncurrent_from_file(readfile,string("n_current/iter_",i))
        Hfluxes[i+1]=(n_current[:H][end]*speciesbcs(:H)[2,2]
                      +2*n_current[:H2][end]*speciesbcs(:H2)[2,2])
    end
    Hfluxes
end

@everywhere function get_rates_and_fluxes(readfile)
    mydset=h5open(readfile,"r")
    mydata=read(mydset)
    timelength=length(mydata["n_current"]["timelist"])+1
    close(mydset)
    reactionrateshist=fill(convert(Float64,NaN),timelength,length(intaltgrid),length(reactionnet))
    fluxhist=fill(convert(Float64,NaN),timelength,length(intaltgrid),length(specieslist))
    n_current=read_ncurrent_from_file(readfile,string("n_current/init"))
    reactionrateshist[1,:,:]=reactionrates(n_current)
    fluxhist[1,:,:]=fluxes(n_current,dz)
    for i in 1:(timelength-1)
        println(readfile,": ",i)
        n_current=read_ncurrent_from_file(readfile,string("n_current/iter_",i))
        reactionrateshist[i+1,:,:]=reactionrates(n_current)
        fluxhist[i+1,:,:]=fluxes(n_current,dz)
    end
    (reactionrateshist,fluxhist)
end

@everywhere function get_all_rates_and_fluxes(readfile)
    (reactionrateshist,fluxhist)=get_rates_and_fluxes(readfile)
    h5write(readfile,"fluxes/flux_history",fluxhist)
    h5write(readfile,"rates/reaction_rates_history",reactionrateshist)
    return
end

pmap(x->println(string("parmsvec[i][1]=",x[1],", parmsvec[i][2]=",x[2],", filename=",x[3])),[[p,f;] for (p,f) in zip(parmsvec,filenamevec)])

##This runs the simulation for a year and returns
## oneyeartimepts=logspace(log10(1),log10(3.14e7),1000)
## oneyeartimediff=oneyeartimepts[2:end]-oneyeartimepts[1:end-1]
## n_converged=get_ncurrent("converged_standardwater.h5")
## n_oneyear=runwaterprofile(n_converged,80,60,oneyeartimediff,"one_year_response_to_80ppm_at_60km.h5")
## n_return=runwaterprofile(n_oneyear,0.,60,oneyeartimediff,"one_year_response_to_80ppm_at_60km_return.h5")

##This runs the simulation for all added ppms and altitudes
## pmap( x->runwaterprofile(n_current,x[1],x[2],timediff,x[3]),[[p,f;] for (p,f) in zip(parmsvec,filenamevec)])
## pmap(get_all_rates_and_fluxes,filenamevec)


## This gets the H fluxes and water profiles for easy figure creation
## Hfluxes=pmap(get_H_fluxes,filenamevec)
## lhfl=length(Hfluxes[1,1])
## writeHfluxes=fill(0.0,(length(waterppmvec),length(wateraltvec),lhfl+2))
## for lp in 1:length(parmsvec)
##     ippm=lp%length(waterppmvec)+1
##     ialt=floor(Int,(lp-1)/length(waterppmvec))+1
##     writeHfluxes[ippm,ialt,:]=[parmsvec[lp],Hfluxes[lp];]
## end
## function get_water_ppm(filename)
##     n_file=read_ncurrent_from_file(filename,"n_current/init")
##     waterppmvec=1e6*n_file[:H2O]./map(z->n_tot(n_file,z),alt[2:end-1])
##     return waterppmvec
## end

## waterprofs=map(get_water_ppm,filenamevec)
## writewaterprof=fill(0.0,(length(waterppmvec),length(wateraltvec),length(alt)-2+2))
## for lp in 1:length(parmsvec)
##     ippm=lp%length(waterppmvec)+1
##     ialt=floor(Int,(lp-1)/length(waterppmvec))+1
##     writewaterprof[ippm,ialt,:]=[parmsvec[lp],waterprofs[lp];]
## end


## h5write("./H_esc_flux_history.h5","fluxes/fluxvals",writeHfluxes)
## h5write("./H_esc_flux_history.h5","fluxes/times",h5read("./ppm_20_alt_20.h5","n_current/timelist"))
## h5write("./H_esc_flux_history.h5","waterprofs/ppm",writewaterprof)
## h5write("./H_esc_flux_history.h5","waterprofs/alt",alt[2:end-1])


##this runs the simulation for removing water from the system, to
##answer the final reviewer objection and show that an imbalance is
##created when water is removed
## n_current[:H2O]=H2Olowfrac.*map(z->n_tot(n_current,z),alt[2:end-1])
## n_final=runprofile(n_current,timediff,"response_to_halving_lower_atmospheric_water.h5")
## get_all_rates_and_fluxes("response_to_halving_lower_atmospheric_water.h5")
## Joke's on me, no imbalance is created!!!

## What about if we zero out upper atmospheric water?
## n_current[:H2O]=H2Onohighalt.*map(z->n_tot(n_current,z),alt[2:end-1])
## n_final=runprofile(n_current,timediff,"response_to_zeroing_upper_atmospheric_water.h5")
## get_all_rates_and_fluxes("response_to_zeroing_upper_atmospheric_water.h5")
## this produces a small change but it's not significant. Only
## removing water from an initially wet upper atmosphere produces net
## oxidation.

