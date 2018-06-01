# This julia code contains a photochemical model of the Martian
# atmosphere prepared by Mike Chaffin, originally for the 8th
# International Conference on Mars, held at CalTech in July 2014.

# Some of the model parameters are stored in global variables (sorry).


#     !!!
#   !!!!!!!
#   !!!!!!!
#   !!!!!!!
#    !!!!!
#     !!!
#
#     !!!
#    !!!!!
#     !!!

###ALL distance/area/volume units must be in CM!!!!

using PyPlot
using HDF5, JLD

#################################
##########Species LIST###########
#################################

#array of symbols for each species
const fullspecieslist = [:CO2, :O2, :O3, :H2, :OH, :HO2, :H2O, :H2O2, :O, :CO,
                         :O1D, :H,:N2,:Ar,:CO2pl,:HOCO];
specieslist=fullspecieslist;
const nochemspecies = [:N2, :Ar, :CO2pl,:H2O];
const chemspecies = setdiff(specieslist, nochemspecies);
const notransportspecies = [:CO2pl,:H2O];
const transportspecies = setdiff(specieslist, notransportspecies);
const speciesmolmasslist = Dict(:CO2=>44, :O2=>32, :O3=>48, :H2=>2, :OH=>9,
                                :HO2=>17, :H2O=>18, :H2O2=>34, :O=>16, :CO=>28,
                                :O1D=>16, :H=>1, :N2=>28, :Ar=>40, :CO2pl=>44,
                                :HOCO=>45)

function fluxsymbol(x)
    Symbol(string("f",string(x)))
end
const fluxlist = map(fluxsymbol, fullspecieslist)

# array of species for which photolysis is important. All rates should
# start with J, end with a lowercase letter, and contain a species in
# specieslist above, which is used to compute photolysis. Different
# letters at the end correspond to different products of photolytic
# destruction or to photoionization.
const Jratelist=[:JCO2ion,:JCO2toCOpO,:JCO2toCOpO1D,:JO2toOpO,:JO2toOpO1D,
                 :JO3toO2pO,:JO3toO2pO1D,:JO3toOpOpO,:JH2toHpH,:JOHtoOpH,
                 :JOHtoO1DpH,:JHO2toOHpH,:JH2OtoHpOH,:JH2OtoH2pO1D,:JH2OtoHpHpO,
                 :JH2O2to2OH,:JH2O2toHO2pH,:JH2O2toH2OpO1D];


##############################################
######Load Converged Test Case from File######
##############################################

# the test case was created by hand by Mike Chaffin and saved for automated use.
# Change following line as needed depending on local machine
readfile = "/home/mike/Documents/Mars/Photochemistry/coupled_transport_and_chemistry/revision2/converged_standardwater.h5"
const alt=h5read(readfile,"n_current/alt")

function get_ncurrent(readfile)
    const alt=h5read(readfile,"n_current/alt")   #TODO: is this redundant?
    n_current_tag_list=map(Symbol, h5read(readfile,"n_current/species"))
    n_current_mat=h5read(readfile,"n_current/n_current_mat");
    n_current=Dict{Symbol, Array{Float64, 1}}()
    for ispecies in [1:length(n_current_tag_list);]
        n_current[n_current_tag_list[ispecies]]=reshape(n_current_mat[:,ispecies],length(alt)-2)
    end
    n_current
end

n_current=get_ncurrent(readfile)

function write_ncurrent(n_current, filename)
    n_current_mat=Array(Float64, length(alt)-2, length(collect(keys(n_current))));
    for ispecies in [1:length(collect(keys(n_current)));]
        for ialt in [1:length(alt)-2;]
            n_current_mat[ialt, ispecies]=n_current[collect(keys(n_current))[ispecies]][ialt]
        end
    end
    h5write(filename,"n_current/n_current_mat",n_current_mat)
    h5write(filename,"n_current/alt",alt)
    h5write(filename,"n_current/species",map(string, collect(keys(n_current))))
end




###########################
#discretization parameters#
###########################

#set altitude grid and transport equilibrium from file
const zmin=alt[1]
const zmax=alt[end];
const dz=alt[2]-alt[1];

# TODO: remove this stuff?
## #ALTITUDE grid
## const zmin=0e5;
## const zmax=200e5;
## const dz=2e5;

## #this altitude grid includes the top and bottom points that are set by
## #the boundary conditions. Numerically, we solve for length(alt)-2
## #densities for each species.
## #Altitudes INCREASE with increasing index.
## const alt=[zmin:dz:zmax];

#initial timestep
dt=360.0;


####################################
##########REACTION NETWORK##########
####################################

#function to replace three body rates with the recommended expression
threebody(k0, kinf) = :($k0*M/(1+$k0*M/$kinf)*0.6^((1+(log10($k0*M/$kinf))^2)^-1))

threebodyca(k0, kinf) = :($k0/(1+$k0/($kinf/M))*0.6^((1+(log10($k0/($kinf*M)))^2)^-1))

reactionnet=[
             #Photodissociation
             [[:CO2],[:CO,:O],:JCO2toCOpO],
             [[:CO2],[:CO,:O1D],:JCO2toCOpO1D],
             [[:O2],[:O,:O],:JO2toOpO],
             [[:O2],[:O,:O1D],:JO2toOpO1D],
             [[:O3],[:O2,:O],:JO3toO2pO],
             [[:O3],[:O2,:O1D],:JO3toO2pO1D],
             [[:O3],[:O,:O,:O],:JO3toOpOpO],
             [[:H2],[:H,:H],:JH2toHpH],
             [[:OH],[:O,:H],:JOHtoOpH],
             [[:OH],[:O1D,:H],:JOHtoO1DpH],
             [[:HO2],[:OH,:O],:JHO2toOHpH], # other branches should be
                                            # here, but have not been
                                            # measured
             [[:H2O],[:H,:OH],:JH2OtoHpOH],
             [[:H2O],[:H2,:O1D],:JH2OtoH2pO1D],
             [[:H2O],[:H,:H,:O],:JH2OtoHpHpO],
             [[:H2O2],[:OH,:OH],:JH2O2to2OH],
             [[:H2O2],[:HO2,:H],:JH2O2toHO2pH],
             [[:H2O2],[:H2O,:O1D],:JH2O2toH2OpO1D],

             # recombination of O
             [[:O,:O,:M],[:O2,:M],:(1.8*3.0e-33*(300/T)^3.25)],
             [[:O,:O2,:N2],[:O3,:N2],:(5e-35*exp(724./T))],
             [[:O,:O2,:CO2],[:O3,:CO2],:(2.5*6.0e-34*(300./T)^2.4)],
             [[:O,:O3],[:O2,:O2],:(8.0e-12*exp(-2060./T))],
             [[:O,:CO,:M],[:CO2,:M],:(2.2e-33*exp(-1780./T))],

             # O1D attack
             [[:O1D,:O2],[:O,:O2],:(3.2e-11*exp(70./T))],
             [[:O1D,:O3],[:O2,:O2],:(1.2e-10)],
             [[:O1D,:O3],[:O,:O,:O2],:(1.2e-10)],
             [[:O1D,:H2],[:H,:OH],:(1.2e-10)],
             [[:O1D,:CO2],[:O,:CO2],:(7.5e-11*exp(115./T))],
             [[:O1D,:H2O],[:OH,:OH],:(1.63e-10*exp(60./T))],

             # loss of H2
             [[:H2,:O],[:OH,:H],:(6.34e-12*exp(-4000/T))],
             [[:OH,:H2],[:H2O,:H],:(9.01e-13*exp(-1526/T))],

             # recombination of H
             [[:H,:H,:CO2],[:H2,:CO2],:(1.6e-32*(298./T)^2.27)],
             [[:H,:OH,:CO2],[:H2O,:CO2],:(1.9*6.8e-31*(300./T)^2)],
             [[:H,:HO2],[:OH,:OH],:(7.2e-11)],
             [[:H,:HO2],[:H2O,:O1D],:(1.6e-12)],#O1D is theoretically mandated
             [[:H,:HO2],[:H2,:O2],:(0.5*6.9e-12)],
             [[:H,:H2O2],[:HO2,:H2],:(2.8e-12*exp(-1890/T))],
             [[:H,:H2O2],[:H2O,:OH],:(1.7e-11*exp(-1800/T))],

             # Interconversion of odd H
             [[:H,:O2],[:HO2],threebody(:(2.0*4.4e-32*(T/300)^-1.3),
                                        :(7.5e-11*(T/300)^0.2))],
             [[:H,:O3],[:OH,:O2],:(1.4e-10*exp(-470/T))],
             [[:O,:OH],[:O2,:H],:(1.8e-11*exp(180/T))],
             [[:O,:HO2],[:OH,:O2],:(3.0e-11*exp(200./T))],
             [[:O,:H2O2],[:OH,:HO2],:(1.4e-12*exp(-2000/T))],
             [[:OH,:OH],[:H2O,:O],:(1.8e-12)],
             [[:OH,:OH],[:H2O2],threebody(:(1.3*6.9e-31*(T/300)^-1.0),:(2.6e-11))],
             [[:OH,:O3],[:HO2,:O2],:(1.7e-12*exp(-940/T))],
             [[:OH,:HO2],[:H2O,:O2],:(4.8e-11*exp(250/T))],
             [[:OH,:H2O2],[:H2O,:HO2],:(1.8e-12)],
             [[:HO2,:O3],[:OH,:O2,:O2],:(1.0e-14*exp(-490/T))],
             [[:HO2,:HO2],[:H2O2,:O2],:(3.0e-13*exp(460/T))],
             [[:HO2,:HO2,:M],[:H2O2,:O2,:M],:(2*2.1e-33*exp(920/T))],

             # CO2 recombination due to odd H (with HOCO intermediate)
             [[:CO,:OH],[:CO2,:H],threebodyca(:(1.5e-13*(T/300)^0.6),:(2.1e9*(T/300)^6.1))],
             [[:OH,:CO],[:HOCO],threebody(:(5.9e-33*(T/300)^-1.4),:(1.1e-12*(T/300)^1.3))],
             [[:HOCO,:O2],[:HO2,:CO2],:(2.0e-12)],

             # CO2+ attack on molecular hydrogen
             [[:CO2pl,:H2],[:CO2,:H,:H],:(8.7e-10)]
             ]

#############################
####FUNDAMENTAL CONSTANTS####
#############################

# fundamental constants
const boltzmannK=1.38e-23;    # J/K
const bigG=6.67e-11;          # N m^2/kg^2
const mH=1.67e-27;            # kg

# mars parameters
const marsM=0.1075*5.972e24;  # kg
const radiusM=3396e5;         # cm

#############################
########TEMPERATURE##########
#############################

function Tspl(z::Float64, lapserate=-1.4e-5, Tsurf=211, ztropo=50e5, zexo=200e5, Texo=300)
    # DO NOT MODIFY! If you want to change the temperature, define a
    # new function or select different arguments and pass to Temp(z)

    # spline interpolation between adaiabatic lapse rate in lower atm
    # (Zahnle 2008) and isothermal exosphere (300K, typical?)  Too hot
    # at 125km?? should be close to 170 (Krasnopolsky)
    if z < ztropo
        return Tsurf + lapserate*z
    end
    if z > zexo
        return Texo
    end
    Ttropo = Tsurf+lapserate*ztropo
    Tret = ((z - zexo)^2 * (Ttropo*(2*z + zexo - 3*ztropo) +
            lapserate*(z - ztropo)*(zexo - ztropo))
            - Texo*(z - ztropo)^2*(2*z - 3*zexo + ztropo))/(zexo - ztropo)^3
    # I got the above expression from Mathematica; seems to do what I
    # asked it to, matching the temp and lapse rate at the tropopause
    # and the temp with no derivative at the exobase.
end

function Tpiecewise(z::Float64, Tinf=240.0, Ttropo=125.0, lapserate=-1.4e-5, ztropo=90e5, ztropowidth=30e5)
    # DO NOT MODIFY! If you want to change the temperature, define a
    # new function or select different arguments and pass to Temp(z)

    # a piecewise function for temperature as a function of altitude,
    # using Krasnopolsky's 2010 temperatures for altitudes
    # >htropo=90km, fixed at Ttropo=125K between htropo and
    # htropo-htropowidth=60km, and rising at a constant lapse rate
    # (1.4K/km) below.
    if z >= ztropo
        return Tinf - (Tinf - Ttropo)*exp(-((z-ztropo)^2)/(8e10*Tinf))
    end
    if ztropo > z >= ztropo - ztropowidth
        return Ttropo
    end
    if ztropo-ztropowidth > z
        return Ttropo-lapserate*(ztropo-ztropowidth-z)
    end
end

#If changes to the temperature are needed, they should be made here
Temp(z::Float64) = Tpiecewise(z)


###############################
#####AUX DENSITY FUNCTIONS#####
###############################

n_alt_index=Dict([z=>clamp((i-1),1, length(alt)-2) for (i, z) in enumerate(alt)])
# used in combination with n_current. Gets the index corresponding to a given altitude

function n_tot(n_current, z)
    # get the total number density at this altitude
    thisaltindex = n_alt_index[z]
    return sum( [n_current[s][thisaltindex] for s in specieslist] )
end

# mean molecular mass at a given altitude
function meanmass(n_current, z)
    #find the mean molecular mass at a given altitude
    thisaltindex = n_alt_index[z]

    c = [n_current[sp][thisaltindex] for sp in specieslist]
    m = [speciesmolmasslist[sp] for sp in specieslist]

    return sum(c.*m)/sum(c)
end


#####################################
########BOUNDARY CONDITIONS##########
#####################################


# water saturation vapor pressure, T in K, Psat in mmHg
# (from Washburn 1924)
Psat(T::Float64)=10^(-2445.5646/T+8.2312*log10(T)-0.01677006*T+1.20514e-5*T^2-6.757169)

# convert to H2O satuation vapor pressure:
# 133.3 N/m2 = 1 mmHg, divided by k*T=J=N*m = 1/m3, convert to 1/cm3... 1/cm3 = 10^6/m3
H2Osat = map(x->(1e-6*133.3*Psat(x))/(boltzmannK*x),map(Temp, alt))
H2Osatfrac = H2Osat./map(z->n_tot(n_current, z),alt)
H2Oinitfrac = H2Osatfrac[1:findfirst(H2Osatfrac, minimum(H2Osatfrac))]
H2Oinitfrac = [H2Oinitfrac, fill(minimum(H2Osatfrac),
               length(alt)-2-length(H2Oinitfrac));]
H2Oinitfrac[find(x->x<30e5, alt)]=1e-4
for i in [1:length(H2Oinitfrac);]
    H2Oinitfrac[i] = H2Oinitfrac[i] < H2Osatfrac[i+1] ? H2Oinitfrac[i] : H2Osatfrac[i+1]
end

# TODO: is this redundant with lower plot?
# plot the initial H2O profile
clf()
plot(H2Oinitfrac/1e-6, alt[2:end-1]/1e5, color="blue",linewidth=2)
ylim(0, 100)
xlabel("Volume Mixing Ratio [ppm]")
ylabel("Altitude [km]")


H2Olowfrac = H2Osatfrac[1:findfirst(H2Osatfrac, minimum(H2Osatfrac))]
H2Olowfrac = [H2Olowfrac, fill(minimum(H2Osatfrac),length(alt)-2-length(H2Olowfrac));]
H2Olowfrac[find(x->x<30e5, alt)]=0.5e-4
for i in [1:length(H2Olowfrac);]
    H2Olowfrac[i] = H2Olowfrac[i] < H2Osatfrac[i+1] ? H2Olowfrac[i] : H2Osatfrac[i+1]
end

H2Onohighalt = deepcopy(H2Oinitfrac)
H2Onohighalt[find(x->x>30e5, alt[2:end-1])]=0.0

# add in a detached water vapor layer, which looks kinda like a
# gaussian packet floating at 60km (Maltagliati 2013)
detachedlayer = 1e-6*map(x->80.*exp(-((x-60)/12.5)^2),alt[2:end-1]/1e5)+H2Oinitfrac

# TODO: keep this plot only?
# Plot initial water profile with detatched layer
clf()
plot(H2Oinitfrac/1e-6, alt[2:end-1]/1e5, color="green",linewidth=5)
plot(detachedlayer/1e-6, alt[2:end-1]/1e5, color="red",linewidth=2, linestyle="--")
plot(H2Olowfrac/1e-6, alt[2:end-1]/1e5, color="blue",linewidth=2, linestyle="--")
ylim(0, 100)
xlabel("Volume Mixing Ratio [ppm]")
ylabel("Altitude [km]")
show()

# this computes the total water column in precipitable microns
# n_col(molecules/cm2)/(molecules/mol)*(gm/mol)*(1cc/gm) = (cm)*(10^4μm/cm)
# = precipitable μm
(sum(([1e-4, H2Oinitfrac;]).*map(z->n_tot(n_current, z),alt[1:end-1]))*dz)/6.02e23*18*1e4
(sum(([1e-4, detachedlayer;]).*map(z->n_tot(n_current, z),alt[1:end-1]))*dz)/6.02e23*18*1e4


# effusion velocities need to be computed from the atmospheric temperature at
# the upper boundary:
function effusion_velocity(Texo::Float64, m::Float64)
    lambda = (m*mH*bigG*marsM)/(boltzmannK*Texo*1e-2*(radiusM+zmax))
    ## println("lambda = ",lambda)
    vth=sqrt(2*boltzmannK*Texo/(m*mH))
    ## println("vth = ", vth)
    v = 1e2*exp(-lambda)*vth*(lambda+1)/(2*pi^0.5)
    return v
end

H_effusion_velocity = effusion_velocity(Temp(zmax),1.0)
H2_effusion_velocity = effusion_velocity(Temp(zmax),2.0)

#boundary conditions for each species (mostly from Nair 1994)

#default boundary condition is zero flux at top and bottom.
const speciesbclist=Dict(
               :CO2=>["n" 2.1e17; "f" 0.],
               :Ar=>["n" 2.0e-2*2.1e17; "f" 0.],
               :N2=>["n" 1.9e-2*2.1e17; "f" 0.],
               :H2O=>["n" H2Osat[1]; "f" 0.], # boundary condition does
                                              # not matter if H2O is
                                              # fixed
               :O=>["f" 0.; "f" 1.2e8],
               :H2=>["f" 0.; "v" H2_effusion_velocity],
               :H=>["f" 0.; "v" H_effusion_velocity],
               );

function speciesbcs(species)
    get(speciesbclist,
        species,
        ["f" 0.; "f" 0.])
end


##############################
####DIFFUSION COEFFICIENTS####
##############################

function Keddy(n_current, z)
    # eddy diffusion coefficient, stolen from Krasnopolsky (1993).
    # Scales as the inverse sqrt of atmospheric number density
    z <= 60.e5 ? 10^6 : 2e13/sqrt(n_tot(n_current, z))
end

function Keddy(z::Real, nt::Real)
    # eddy diffusion coefficient, stolen from Krasnopolsky (1993).
    # Scales as the inverse sqrt of atmospheric number density
    z <= 60.e5 ? 10^6 : 2e13/sqrt(nt)
end

# molecular diffusion parameters. molecular diffusion is different only
# for small molecules and atoms (H and H2), otherwise all species share
# the same values (Krasnopolsky 1993 <- Banks and Kockarts Aeronomy)
# THESE ARE IN cm^-2 s^-2!!!
diffparams(species) = get(Dict(:H=>[8.4, 0.597],:H2=>[2.23, 0.75]),species,[1.0, 0.75])
function Dcoef(T, n::Real, species::Symbol)
    dparms = diffparams(species)
    dparms[1]*1e17*T^(dparms[2])/n
end
Dcoef(z, species::Symbol, n_current) = Dcoef(Temp(z),n_tot(n_current, z),species)

#thermal diffusion factors (from Krasnopolsky 2002)
thermaldiff(species) = get(Dict(:H=>-0.25,:H2=>-0.25,:HD=>-0.25,:D2=>-0.25,
                                :He=>-0.25), species, 0)

###############################
##########TRANSPORT############
###############################

# scale height at a given altitude
function scaleH(z, T::Float64, mm::Real)
    boltzmannK*T/(mm*mH*marsM*bigG)*(((z+radiusM)*1e-2)^2)*1e2
    # constants are in MKS. Convert to m and back to cm.
end

function scaleH(z, species::Symbol)
    T=Temp(z)
    mm = speciesmolmasslist[species]
    scaleH(z, T, mm)
end

function scaleH(z, T::Float64, species::Symbol)
    mm = speciesmolmasslist[species]
    scaleH(z, T, mm)
end

function scaleH(z, T::Float64, n_current)
    mm = meanmass(n_current, z)
    scaleH(z, T, mm)
end

function scaleH(z, n_current)
    T=Temp(z)
    scaleH(z, T)
end

# at each level of the atmosphere, density can be transferred up or
# down with a specified rate coefficient.
#
#                          | n_i+1
#       ^  tspecies_i_up   v tspecies_i+1_down
#   n_i |
#       v tspecies_i_down  ^ tspecies_i-1_up
#                          | n_i-1
#
#
# the flux at each cell boundary is the sum of the upward flux from
# the cell below and the downward flux of the cell above. These fluxes
# are determined using flux coefficients that come from the diffusion
# equation. Care must be taken at the upper and lower boundary so that
# tspecies_top_up and tspecies_bottom_down properly reflect the
# boundary conditions of the atmosphere.


# This is handled in the code with the population of the appropriate
# reactions, with variable rate coeffecients that are populated
# between each timestep (similar to the way photolysis rates are
# included). We need to make reactions at each interior altitude
# level:
#          n_i -> n_i+1  tspecies_i_up
#          n_i -> n_i-1  tspecies_i_down
#
# At the upper and lower boundary we omit the species on the RHS, so
# that these reactions are potentially non-conservative:
#
#         n_top    -> NULL  tspecies_top_up
#         n_bottom -> NULL  tspecies_bottom_down
#
# These coefficients then describe the diffusion velocity at the top
# and bottom of ther atmosphere.


# function to generate coefficients of the transport network
function fluxcoefs(z, dz, ntv, Kv, Dv, Tv, Hsv, H0v, species)
    Dl = (Dv[1] + Dv[2])/2.0
    Kl = (Kv[1] + Kv[2])/2.0
    Tl = (Tv[1] + Tv[2])/2.0
    dTdzl = (Tv[2] - Tv[1])/dz
    Hsl = (Hsv[1] + Hsv[2])/2.0
    H0l = (H0v[1] + H0v[2])/2.0

    # we have two flux terms to combine:
    sumeddyl = (Dl+Kl)/dz/dz
    gravthermall = (Dl*(1/Hsl + (1+thermaldiff(species))/Tl*dTdzl) +
                    Kl*(1/H0l + 1/Tl*dTdzl))/(2*dz)

    Du = (Dv[2] + Dv[3])/2.0
    Ku = (Kv[2] + Kv[3])/2.0
    Tu = (Tv[2] + Tv[3])/2.0
    dTdzu = (Tv[3] - Tv[2])/dz
    Hsu = (Hsv[2] + Hsv[3])/2.0
    H0u = (H0v[2] + H0v[3])/2.0

    # we have two flux terms to combine:
    sumeddyu = (Du+Ku)/dz/dz
    gravthermalu = (Du*(1/Hsu + (1 + thermaldiff(species))/Tu*dTdzu) +
                  Ku*(1/H0u + 1/Tu*dTdzu))/(2*dz)

    # this results in the following coupling coefficients:
    return [sumeddyl+gravthermall, # down
            sumeddyu-gravthermalu] # up
end

# overload to generate the coefficients if they are not supplied
function fluxcoefs(z, dz, species, n_current)
    ntp = n_tot(n_current, z+dz)
    nt0 = n_tot(n_current, z)
    ntm = n_tot(n_current, z-dz)
    Kp = Keddy(z+dz, ntp)
    K0 = Keddy(z, nt0)
    Km = Keddy(z-dz, ntm)
    Tp = Temp(z+dz)
    T0 = Temp(z)
    Tm = Temp(z-dz)
    Dp = Dcoef(Tp, ntp, species)
    D0 = Dcoef(T0, nt0, species)
    Dm = Dcoef(Tm, ntm, species)
    Hsp = scaleH(z+dz, species)
    Hs0 = scaleH(z, species)
    Hsm = scaleH(z-dz, species)
    H0p = scaleH(z+dz, Tp, n_current)
    H00 = scaleH(z, T0, n_current)
    H0m = scaleH(z-dz, Tm, n_current)

    # return the coefficients
    return fluxcoefs(z, dz,
              [ntm, nt0, ntp],
              [Km , K0, Kp],
              [Dm , D0, Dp],
              [Tm , T0, Tp],
              [Hsm, Hs0, Hsp],
              [H0m, H00, H0p],
              species)
end

# define transport coefficients for a given atmospheric layers for transport
# from a layer to the one above
function lower_up(z, dz, species, n_current)
    ntp = n_tot(n_current, z+dz)
    nt0 = n_tot(n_current, z)
    ntm = 1
    Kp = Keddy(z+dz, ntp)
    K0 = Keddy(z,nt0)
    Km = 1
    Tp = Temp(z+dz)
    T0 = Temp(z)
    Tm = 1
    Dp = Dcoef(Tp, ntp, species)
    D0 = Dcoef(T0, nt0, species)
    Dm = 1
    Hsp = scaleH(z+dz, species)
    Hs0 = scaleH(z,species)
    Hsm = 1
    H0p = scaleH(z+dz, Tp, n_current)
    H00 = scaleH(z,T0, n_current)
    H0m = 1

    #return the coefficients
    return fluxcoefs(z, dz,
              [ntm, nt0, ntp],
              [Km , K0, Kp],
              [Dm , D0, Dp],
              [Tm , T0, Tp],
              [Hsm, Hs0, Hsp],
              [H0m, H00, H0p],
              species)[2]
end

# define transport coefficients for a given atmospheric layers for transport
# from a layer to the one below
function upper_down(z, dz, species, n_current)
    ntp = 1
    nt0 = n_tot(n_current, z)
    ntm = n_tot(n_current, z-dz)
    Kp = 1
    K0 = Keddy(z, nt0)
    Km = Keddy(z-dz, ntm)
    Tp = 1
    T0 = Temp(z)
    Tm = Temp(z-dz)
    Dp = 1
    D0 = Dcoef(T0, nt0, species)
    Dm = Dcoef(Tm, ntm, species)
    Hsp = 1
    Hs0 = scaleH(z, species)
    Hsm = scaleH(z-dz, species)
    H0p = 1
    H00 = scaleH(z, T0, n_current)
    H0m = scaleH(z-dz, Tm, n_current)

    #return the coefficients
    return fluxcoefs(z, dz,
              [ntm, nt0, ntp],
              [Km , K0, Kp],
              [Dm , D0, Dp],
              [Tm , T0, Tp],
              [Hsm, Hs0, Hsp],
              [H0m, H00, H0p],
              species)[1]
end

function boundaryconditions(species, dz, n_current)
    # returns the symbolic transport coefficients that encode the
    # boundary conditions for the null-pointing equations

    # n_1->NULL t_lower_bc
    # n_f->NULL t_upper_bc


    # this defines two additional symbols for each species that need
    # to be resolved in the function call macro:
    #            tspecies_lower_up and
    #            tspecies_upper_down
    # these are found by passing the appropriate values to fluxcoefs
    # and selecting the correct output

    bcs = speciesbcs(species)
    if issubset([species],notransportspecies)
        bcs = ["f" 0.; "f" 0.]
    end

    #first element returned corresponds to lower BC, second to upper
    #BC transport rate. Within each element, the two rates correspond
    #to the two equations
    # n_b  -> NULL (first rate, depends on species concentration)
    # NULL -> n_b  (second rate, independent of species concentration
    bcvec = Float64[0 0;0 0]

    # LOWER
    if bcs[1, 1] == "n"
        bcvec[1,:]=[fluxcoefs(alt[2], dz, species, n_current)[1],
                    lower_up(alt[1], dz, species, n_current)*bcs[1, 2]]
    elseif bcs[1, 1] == "f"
        bcvec[1,:] = [0.0, bcs[1, 2]/dz]
    elseif bcs[1, 1] == "v"
        bcvec[1,:] = [bcs[1, 2]/dz, 0.0]
    else
        throw("Improper lower boundary condition!")
    end

    # UPPER
    if bcs[2, 1] == "n"
        bcvec[2,:] = [fluxcoefs(alt[end-1],dz, species, n_current)[2],
                    upper_down(alt[end],dz, species, n_current)*bcs[1, 2]]
    elseif bcs[2, 1] == "f"
            bcvec[2,:] = [0.0,-bcs[2, 2]/dz]
    elseif bcs[2, 1] == "v"
        bcvec[2,:] = [bcs[2, 2]/dz, 0.0]
    else
        throw("Improper upper boundary condition!")
    end

    #return the bc vec
    return bcvec
end


########################
#######CHEMISTRY########
########################

# TODO: again, the Any[] syntax might be removable once [[],[]] concatenation
# is diabled rather than depracated
function getpos(array, test::Function, n=Any[])
    # this function searches through an arbitrarily structured array
    # finding elements that match the test function supplied, and returns a
    # one-dimensional array of the indicies of these elements.
    if !isa(array, Array)
        test(array) ? Any[n] : Any[]
    else
        vcat([ getpos(array[i], test, Any[n...,i]) for i=1:size(array)[1] ]...)
    end
end

function getpos(array, value)
    # overloading getpos for the most common use case, finding indicies
    # corresponding to elements in array that match value.
    getpos(array, x->x==value)
end

function deletefirst(A, v)
    # Returns list A with its first element equal to v removed.
    index = findfirst(A, v)
    keep = setdiff([1:length(A);],index)
    A[keep]
end

function loss_equations(network, species)
    # given a network of equations in the form of reactionnet above, this
    # function returns the LHS (reactants) and rate coefficient for all
    # reactions where the supplied species is consumed.

    # get list of all chemical reactions species participates in:
    speciespos = getpos(network, species)
    # find pos where species is on LHS but not RHS:
    lhspos = map(x->x[1],  # we only need the reaction number
               map(x->speciespos[x],  # select the appropriate reactions
                   find(x->x[2]==1, speciespos)))
    rhspos = map(x->x[1],  # we only need the reaction number
               map(x->speciespos[x],  # select the appropriate reactions
                   find(x->x[2]==2, speciespos)))
    for i in intersect(lhspos, rhspos)
        lhspos = deletefirst(lhspos, i)
    end

    # get the products and rate coefficient for the identified
    # reactions.
    losseqns=map(x->vcat(Any[network[x][1]...,network[x][3]]), lhspos)
    # automatically finds a species where it occurs twice on the LHS
end

function loss_rate(network, species)
    # return a symbolic expression for the loss rate of species in the
    # supplied reaction network.
    leqn=loss_equations(network, species) # get the equations
    lval=:(+($( # and add the products together
               map(x->:(*($(x...))) # take the product of the
                                    # concentrations and coefficients
                                    # for each reaction
                   ,leqn)...)))
end

function production_equations(network, species)
    # given a network of equations in the form of reactionnet above, this
    # function returns the LHS (reactants) and rate coefficient for all
    # reactions where the supplied species is a product.
    speciespos = getpos(network, species)#list of all reactions where species is produced
    # find pos where species is on RHS but not LHS
    lhspos = map(x->x[1], # we only need the reaction number
               map(x->speciespos[x], #s elect the appropriate reactions
                   find(x->x[2]==1, speciespos)))
    rhspos = map(x->x[1], # we only need the reaction number
               map(x->speciespos[x], # select the appropriate reactions
                   find(x->x[2]==2, speciespos)))
    for i in intersect(rhspos, lhspos)
        rhspos=deletefirst(rhspos, i)
    end

    prodeqns = map(x->vcat(Any[network[x][1]...,network[x][3]]), # get the products and rate
                                                            # coefficient for the identified
                                                            # reactions.
                 # automatically finds and counts duplicate
                 # production for each molecule produced
                 rhspos)

    return prodeqns
end

function production_rate(network, species)
    # return a symbolic expression for the loss rate of species in the
    # supplied reaction network.

    # get the reactants and rate coefficients
    peqn = production_equations(network, species)

    # and add them all up
    # take the product of each set of reactants and coeffecient
    pval = :(+ ( $(map(x -> :(*($(x...))), peqn) ...) ))
end

function chemical_jacobian(chemnetwork, transportnetwork, specieslist, dspecieslist)
    # Compute the symbolic chemical jacobian of a supplied reaction network
    # for the specified list of species. Returns three arrays suitable for
    # constructing a sparse matrix: lists of the first and second indices
    # and the symbolic value to place at that index.

    # set up output vectors: indices and values
    ivec = Int64[] # list of first indices (corresponding to the species being produced and lost)
    jvec = Int64[] # list of second indices (corresponding to the derivative being taken)
    tvec = Any[] # list of the symbolic values corresponding to the jacobian

    nspecies = length(specieslist)
    ndspecies = length(dspecieslist)
    for i in 1:nspecies #for each species
        ispecies = specieslist[i]
        #get the production and loss equations
        peqn = []
        leqn = []
        if issubset([ispecies],chemspecies)
            peqn = [peqn, production_equations(chemnetwork, ispecies);]
            leqn = [leqn, loss_equations(chemnetwork, ispecies);]
        end
        if issubset([ispecies],transportspecies)
            peqn = [peqn, production_equations(transportnetwork, ispecies);]
            leqn = [leqn, loss_equations(transportnetwork, ispecies);]
        end
        for j in 1:ndspecies #now take the derivative with resp`ect to the other species
            jspecies = dspecieslist[j]
            #find the places where the production rates depend on
            #jspecies, and return the list rates with the first
            #occurrance of jspecies deleted. (Note: this seamlessly
            #deals with multiple copies of a species on either side of
            #an equation, because it is found twice wherever it lives)
            ppos = map(x->deletefirst(peqn[x[1]],jspecies),getpos(peqn, jspecies))
            lpos = map(x->deletefirst(leqn[x[1]],jspecies),getpos(leqn, jspecies))
            if length(ppos)+length(lpos)>0 #if there is a dependence
                #make note of where this dependency exists
                append!(ivec,[i])
                append!(jvec,[j])
                #smash the production and loss rates together,
                #multiplying for each distinct equation, adding
                #together the production and loss seperately, and
                #subtracting loss from production.
                if length(ppos)==0
                    lval = :(+($(map(x->:(*($(x...))),lpos)...)))
                    tval = :(-($lval))
                elseif length(lpos)==0
                    pval = :(+($(map(x->:(*($(x...))),ppos)...)))
                    tval = :(+($pval))
                else
                    pval = :(+($(map(x->:(*($(x...))),ppos)...)))
                    lval = :(+($(map(x->:(*($(x...))),lpos)...)))
                    tval = :(-($pval,$lval))
                end
                #attach the symbolic expression to the return values
                append!(tvec,[tval])
            end
        end
    end
    return (ivec, jvec, Expr(:vcat, tvec...))
end

################################################
########COMBINED CHEMISTRY AND TRANSPORT########
################################################

# We now have objects that return the list of indices and coefficients
# for transport, assuming no other species in the atmosphere
# (transportmat), and for chemistry, assuming no other altitudes
# (chemical_jacobian). We need to perform a kind of outer product on
# these operators, to determine a fully coupled set of equations for
# all species at all altitudes.

# need to get a list of all species at all altitudes to iterate over
const intaltgrid=round.(Int64, alt/1e5)[2:end-1]
const replacespecies=[fullspecieslist, Jratelist,[:T,:M];]

# the rates at each altitude can be computed using the reaction network
# already in place, plus additional equations describing the transport
# to and from the cells above and below:

upeqns = [Any[Any[[s], [Symbol(string(s)*"_above")],Symbol("t"*string(s)*"_up")],
         Any[[Symbol(string(s)*"_above")],[s],Symbol("t"*string(s)*"_above_down")]]
        for s in specieslist]

downeqns = [Any[Any[[s], [Symbol(string(s)*"_below")],Symbol("t"*string(s)*"_down")],
           Any[[Symbol(string(s)*"_below")],[s],Symbol("t"*string(s)*"_below_up")]]
          for s in specieslist]

# NOTE: next line works but is really ugly. It is no longer possible to transpose
# symbol arrays in Julia 0.6, and permutedims doesn't work either.
local_transport_rates = [[[Symbol("t"*string(s)*"_up") for s in specieslist]
                          [Symbol("t"*string(s)*"_down") for s in specieslist]
                          [Symbol("t"*string(s)*"_above_down") for s in specieslist]
                          [Symbol("t"*string(s)*"_below_up") for s in specieslist]]...;]


transportnet = [[upeqns...;],[downeqns...;];]

# define names for all the species active in the coupled rates:
activespecies = union(chemspecies, transportspecies)
active_above = [Symbol(string(s)*"_above") for s in activespecies]
active_below = [Symbol(string(s)*"_below") for s in activespecies]

inactivespecies = intersect(nochemspecies, notransportspecies)

function getrate(chemnet, transportnet, species)
    # Calculates the rate at which a given species is either produced or lost.
    # Production is from chemical reaction yields or entry from other
    # atmospheric layers. Loss is due to consumption in reactions or migration
    # to other layers. Comment added by EMC 8/29/17
    rate = :(0.0)
    if issubset([species],chemspecies)
        rate = :($rate
               + $(production_rate(chemnet, species))
               - $(      loss_rate(chemnet, species)))
    end
    if issubset([species],transportspecies)
        rate = :($rate
               + $(production_rate(transportnet, species))
               - $(      loss_rate(transportnet, species)))
    end

    return rate
end

# obtain the rates and jacobian for each altitude
const rates_local = Expr(:vcat, map(x->getrate(reactionnet, transportnet, x),activespecies)...);
const chemJ_local = chemical_jacobian(reactionnet, transportnet, activespecies, activespecies);
const chemJ_above = chemical_jacobian(reactionnet, transportnet, activespecies, active_above);
const chemJ_below = chemical_jacobian(reactionnet, transportnet, activespecies, active_below);

arglist_local = [activespecies,
                 active_above,
                 active_below,
                 inactivespecies,
                 Jratelist,
                 :T, :M,
                 local_transport_rates,
                 :dt;]

arglist_local_typed=[:($s::Float64) for s in arglist_local]

@eval begin
    function ratefn_local($(arglist_local_typed[1:end-1]...))
        $rates_local
    end
end

@eval begin
    function chemJmat_local($(arglist_local_typed...))
        localchemJi = $(chemJ_local[1])
        localchemJj = $(chemJ_local[2])
        localchemJval = -dt*$(chemJ_local[3])

        abovechemJi = $(chemJ_above[1])
        abovechemJj = $(chemJ_above[2])
        abovechemJval = -dt*$(chemJ_above[3])

        belowchemJi = $(chemJ_below[1])
        belowchemJj = $(chemJ_below[2])
        belowchemJval = -dt*$(chemJ_below[3])

        ((localchemJi, localchemJj, localchemJval),
         (abovechemJi, abovechemJj, abovechemJval),
         (belowchemJi, belowchemJj, belowchemJval))
    end
end

# a function to return chemical reaction rates for specified species
# concentrations
@eval begin
    function reactionrates_local($(specieslist...), $(Jratelist...), T,  M)
        $(Expr(:vcat, map(x->Expr(:call,:*,x[1]..., x[3]), reactionnet)...))
    end
end


function reactionrates(n_current)
    theserates = fill(convert(Float64, NaN),(length(intaltgrid),length(reactionnet)))
    for ialt in 1:length(intaltgrid)
        theserates[ialt,:] = reactionrates_local([[n_current[sp][ialt] for sp in specieslist],
                                                [n_current[J][ialt] for J in Jratelist],
                                                Temp(alt[ialt+1]),
                                                n_tot(n_current, alt[ialt+1]);]...)
    end
    return theserates
end



function getflux(n_current, dz, species)
    thesecoefs = [fluxcoefs(a, dz, species, n_current) for a in alt[2:end-1]]
    thesebcs = boundaryconditions(species, dz, n_current)

    thesefluxes = fill(convert(Float64, NaN),length(intaltgrid))

    thesefluxes[1] = (-(n_current[species][2]*thesecoefs[2][1]
                      -n_current[species][1]*thesecoefs[1][2])
                    +(-n_current[species][1]*thesebcs[1, 1]
                      +thesebcs[1, 2]))/2.0
    for ialt in 2:length(intaltgrid)-1
        thesefluxes[ialt] = (-(n_current[species][ialt+1]*thesecoefs[ialt+1][1]
                             - n_current[species][ialt]*thesecoefs[ialt][2])
                             + (-n_current[species][ialt]*thesecoefs[ialt][1]
                             + n_current[species][ialt-1]*thesecoefs[ialt-1][2]))/2.0
    end
    thesefluxes[end] = (-(thesebcs[2, 2]
                        - n_current[species][end]*thesebcs[2, 1])
                        + (-n_current[species][end]*thesecoefs[end][1]
                        + n_current[species][end-1]*thesecoefs[end-1][2]))/2.0
    return dz*thesefluxes
end

function fluxes(n_current, dz)
    thesefluxes = fill(convert(Float64, NaN),(length(intaltgrid),length(specieslist)))
    for isp in 1:length(specieslist)
        thesefluxes[:,isp] = getflux(n_current, dz, specieslist[isp])
    end
    thesefluxes
end


function ratefn(nthis, inactive, Jrates, T, M, tup, tdown, tlower, tupper)
    # at each altitude, get the appropriate group of concentrations,
    # coefficients, and rates to pass to ratefn_local
    nthismat = reshape(nthis, (length(activespecies), length(intaltgrid)))
    inactivemat = reshape(inactive,(length(inactivespecies),length(intaltgrid)))

    returnrates = zeros(nthismat)

    # fill the first altitude entry with information for all species
    returnrates[:,1] = ratefn_local([nthismat[:,1], nthismat[:,2],
                                    fill(1.0, length(activespecies)),
                                    inactivemat[:,1], Jrates[:,1], T[1],M[1],
                                    tup[:,1], tlower[:,1], tdown[:,2],
                                    tlower[:,2];]...)

    # iterate through other altitudes except the last level, filling the info in
    for ialt in 2:(length(intaltgrid)-1)
        returnrates[:,ialt] = ratefn_local([nthismat[:,ialt],
                                          nthismat[:,ialt+1],
                                          nthismat[:,ialt-1],
                                          inactivemat[:,ialt],
                                          Jrates[:,ialt],
                                          T[ialt], M[ialt],
                                          tup[:,ialt],tdown[:,ialt],
                                          tdown[:,ialt+1],tup[:,ialt-1];]...)
    end

    # fill in the last level of altitude (200 km)
    returnrates[:,end] = ratefn_local([nthismat[:,end],
                                       fill(1.0, length(activespecies)),
                                       nthismat[:,end-1],
                                       inactivemat[:,end],
                                       Jrates[:,end],
                                       T[end], M[end],
                                       tupper[:,1], tdown[:,end],
                                       tupper[:,2], tup[:,end-1];]...)
    return [returnrates...;]
end


function chemJmat(nthis, inactive, Jrates, T, M, tup, tdown, tlower, tupper, dt)
    nthismat = reshape(nthis, (length(activespecies), length(intaltgrid)))
    inactivemat = reshape(inactive, (length(inactivespecies), length(intaltgrid)))
    chemJi = Int64[]
    chemJj = Int64[]
    chemJval = Float64[]
    (tclocal, tcupper, tclower) = chemJmat_local([nthismat[:,1],
                                              nthismat[:,2],
                                              fill(1.0, length(activespecies)),
                                              inactivemat[:,1],
                                              Jrates[:,1],
                                              T[1], M[1],
                                              tup[:,1], tlower[:,1],
                                              tdown[:,2], tlower[:,2],dt;]...)
    #add the influence of the local densities
    append!(chemJi, tclocal[1])
    append!(chemJj, tclocal[2])
    append!(chemJval, tclocal[3])
    #and the upper densities
    append!(chemJi, tcupper[1])
    append!(chemJj, tcupper[2]+length(activespecies))
    append!(chemJval, tcupper[3])

    for ialt in 2:(length(intaltgrid)-1)
        (tclocal, tcupper, tclower) = chemJmat_local([nthismat[:,ialt],
                                                      nthismat[:,ialt+1],
                                                      nthismat[:,ialt-1],
                                                      inactivemat[:,ialt],
                                                      Jrates[:,ialt],
                                                      T[ialt], M[ialt],
                                                      tup[:,ialt],tdown[:,ialt],
                                                      tdown[:,ialt+1],
                                                      tup[:,ialt-1],dt;]...)
        #add the influence of the local densities
        append!(chemJi, tclocal[1]+(ialt-1)*length(activespecies))
        append!(chemJj, tclocal[2]+(ialt-1)*length(activespecies))
        append!(chemJval, tclocal[3])
        #and the upper densities
        append!(chemJi, tcupper[1]+(ialt-1)*length(activespecies))
        append!(chemJj, tcupper[2]+(ialt  )*length(activespecies))
        append!(chemJval, tcupper[3])
        #and the lower densities
        append!(chemJi, tclower[1]+(ialt-1)*length(activespecies))
        append!(chemJj, tclower[2]+(ialt-2)*length(activespecies))
        append!(chemJval, tclower[3])
    end

    (tclocal, tcupper, tclower)=chemJmat_local([nthismat[:,end],
                                              fill(1.0, length(activespecies)),
                                              nthismat[:,end-1],
                                              inactivemat[:,end],
                                              Jrates[:,end],
                                              T[end], M[end],
                                              tupper[:,1], tdown[:,end],
                                              tupper[:,2], tup[:,end-1],dt;]...)
    #add the influence of the local densities
    append!(chemJi, tclocal[1]+(length(intaltgrid)-1)*length(activespecies))
    append!(chemJj, tclocal[2]+(length(intaltgrid)-1)*length(activespecies))
    append!(chemJval, tclocal[3])
    #and the lower densities
    append!(chemJi, tclower[1]+(length(intaltgrid)-1)*length(activespecies))
    append!(chemJj, tclower[2]+(length(intaltgrid)-2)*length(activespecies))
    append!(chemJval, tclower[3])

    #make sure to add 1's along the diagonal
    append!(chemJi,[1:length(nthis);])
    append!(chemJj,[1:length(nthis);])
    append!(chemJval, fill(1.0, length(nthis)))

    sparse(chemJi,
           chemJj,
           chemJval,
           length(nthis),
           length(nthis),
           +);

end

function next_timestep(nstart::Array{Float64, 1},
                       nthis::Array{Float64, 1},
                       inactive::Array{Float64, 1},
                       Jrates::Array{Float64, 2},
                       T::Array{Float64, 1},
                       M::Array{Float64, 1},
                       tup::Array{Float64, 2},
                       tdown::Array{Float64, 2},
                       tlower::Array{Float64, 2},
                       tupper::Array{Float64, 2},
                       dt::Float64)
   # moves to the next timestep using Newton's method on the linearized
   # coupled transport and chemical reaction network.

    eps = 1.0#ensure at least one iteration
    iter = 0
    while eps>1e-8
        nold = deepcopy(nthis)

        #stuff concentrations into update function and jacobian
        fval = nthis - nstart - dt*ratefn(nthis, inactive, Jrates, T, M, tup,
                                          tdown, tlower, tupper)
        #if eps>1e-2; updatemat=chemJmat([nthis, nochems, phrates, T, M, dt]...); end;

        updatemat = chemJmat(nthis, inactive, Jrates, T, M, tup, tdown, tlower,
                             tupper, dt)

        # update
        nthis = nthis - (updatemat \ fval)
        #check relative size of update
        eps = maximum(abs.(nthis-nold)./nold)
        #println("eps=",eps)
        iter += 1
        if iter>1e3; throw("too many iterations in next_timestep!"); end;
    end
    return nthis
end


@everywhere function update!(n_current::Dict{Symbol, Array{Float64, 1}},dt)
    # update n_current using the coupled reaction network, moving to
    # the next timestep

    #set auxiliary (not solved for in chemistry) species values
    inactive = deepcopy(Float64[[n_current[sp][ialt] for sp in inactivespecies, ialt in 1:length(intaltgrid)]...])

    #set photolysis rates
    Jrates = deepcopy(Float64[n_current[sp][ialt] for sp in Jratelist, ialt in 1:length(intaltgrid)])

    #extract concentrations
    nstart = deepcopy([[n_current[sp][ialt] for sp in activespecies, ialt in 1:length(intaltgrid)]...])
    M = sum([n_current[sp] for sp in fullspecieslist])

    # TODO: Is is really necessary to do this every time update! is run?
    # set temperature and total atmospheric concentration
    T = Float64[Temp(a) for a in alt[2:end-1]]

    # take initial guess
    nthis = deepcopy(nstart)

    # get the transport rates
    tup = Float64[issubset([sp],notransportspecies) ? 0.0 : fluxcoefs(a, dz, sp, n_current)[2] for sp in specieslist, a in alt[2:end-1]]
    tdown = Float64[issubset([sp],notransportspecies) ? 0.0 : fluxcoefs(a, dz, sp, n_current)[1] for sp in specieslist, a in alt[2:end-1]]

    # put the lower bcs and upper bcs in separate arrays; but they are not the
    # right shape!
    tlower_temp = [boundaryconditions(sp, dz, n_current)[1,:] for sp in specieslist]
    tupper_temp = [boundaryconditions(sp, dz, n_current)[2,:] for sp in specieslist]

    # reshape tlower and tupper into 2x2 arrays
    tlower = zeros(Float64, length(tlower_temp), 2)
    tupper = zeros(Float64, length(tupper_temp), 2)

    # tlower_temp & tupper_temp have same length; OK to use lower for the range
    for r in range(1, length(tlower_temp))
        tlower[r, :] = tlower_temp[r]
        tupper[r, :] = tupper_temp[r]
    end

    # update to next timestep
    nthis = next_timestep(nstart, nthis, inactive, Jrates, T, M, tup, tdown,
                          tlower, tupper, dt)
    nthismat = reshape(nthis,(length(activespecies),length(intaltgrid)))

    # write found values out to n_current
    for s in 1:length(activespecies)
        for ia in 1:length(intaltgrid)
            tn = nthismat[s, ia]
            n_current[activespecies[s]][ia] = tn > 0. ? tn : 0.
        end
    end

    update_Jrates!(n_current)

end #update!

speciescolor =Dict(
   :HOCO => "#dead91",
   :H2O2 => "#bebddb",
   :HO2 => "#9e9ac8",
   :O3 => "#83d0c1",
   :OH => "#b593e2",
   :H2O => "#1f78b4",
   :O1D => "#269e56",
   :CO2pl => "#eed2d4",
   :H => "#ef3b2c",
   :H2 => "#fc9aa1",
   :O2 => "#41ae76",
   :O => "#00702d",
   :CO => "#fd8d3c",
   :Ar => "#808080",
   :N2 => "#cccccc",
   :CO2 => "#fdd0a2"
   );


function plotatm(n_current)
    clf()
    for sp in fullspecieslist
        plot(n_current[sp], alt[2:end-1]/1e5, color = speciescolor[sp],
             linewidth = 4)
    end
    ylim(0, 200)
    xscale("log")
    xlim(1e-15, 1e18)
    xticks(size = 40)
    yticks(size = 40)
    xlabel("Species concentration [/cm3]",size = 40)
    ylabel("Altitude [km]",size=40)
end

function plotatm()
    plotatm(n_current)
end


####################################
####PHOTOCHEMICAL CROSS SECTIONS####
####################################

# Change following line as needed depending on local machine
xsecfolder="/home/mike/Documents/Mars/Photochemistry/Photochemical Data/From_Justin/uv/uvxsect/";

function readandskip(a, delim::Char, T::Type; skipstart=0)
    # function to read in data from a file, skipping zero or more lines at
    # the beginning.
    a = open(a,"r")
    if skipstart>0
        for i in [1:skipstart;]
            readline(a)
        end
    end
    a = readdlm(a, delim, T)
end


#CO2, temperature-dependent between 195-295K
co2xdata = readandskip(xsecfolder*"CO2.dat",'\t',Float64, skipstart = 4)
function co2xsect(T::Float64)
    clamp(T, 195, 295)
    Tfrac = (T-195)/(295-195)

    arr = [co2xdata[:,1], (1-Tfrac)*co2xdata[:,2]+Tfrac*co2xdata[:,3];]
    reshape(arr, length(co2xdata[:,1]),2)
end

#CO2 photoionization (used to screen high energy sunlight)
co2exdata = readandskip(xsecfolder*"binnedCO2e.csv",',',Float64, skipstart = 4)

#H2O
h2oxdata = readandskip(xsecfolder*"h2oavgtbl.dat",'\t',Float64, skipstart = 4)

#H2O2
#the data in the table cover the range 190-260nm
h2o2xdata = readandskip(xsecfolder*"H2O2.dat",'\t',Float64, skipstart = 3)
#from 260-350 the following analytic calculation fitting the
#temperature dependence is recommended:
function h2o2xsect_l(l::Float64, T::Float64)
    l = clamp(l, 260, 350)
    T = clamp(T, 200, 400)

    A = [64761., -921.70972, 4.535649,
       -0.0044589016, -0.00004035101,
       1.6878206e-7, -2.652014e-10, 1.5534675e-13]
    B = [6812.3, -51.351, 0.11522, -0.000030493, -1.0924e-7]

    lpowA = map(n->l^n,[0:7;])
    lpowB = map(n->l^n,[0:4;])

    expfac = 1.0/(1+exp(-1265/T))

    1e-21*(expfac*reduce(+,map(*,A, lpowA))+(1-expfac)*reduce(+,map(*,B, lpowB)))
end

function h2o2xsect(T::Float64)
    retl = h2o2xdata[:,1]
    retx = 1e4*h2o2xdata[:,2]#factor of 1e4 b/c file is in 1/m2
    addl = [260.5:349.5;]
    retl = [retl, addl;]
    retx = [retx, map(x->h2o2xsect_l(x, T),addl);]
    reshape([retl, retx;],length(retl),2)
end



#Ozone, including IR bands which must be resampled from wavenumber
o3xdata = readandskip(xsecfolder*"O3.dat",'\t',Float64, skipstart=3)
o3ls = o3xdata[:,1]
o3xs = o3xdata[:,2]
o3chapxdata = readandskip(xsecfolder*"O3Chap.dat",'\t',Float64, skipstart=3)
o3chapxdata[:,1] = map(p->1e7/p, o3chapxdata[:,1])
for i in [round(Int, floor(minimum(o3chapxdata[:,1]))):round(Int, ceil(maximum(o3chapxdata))-1);]
    posss = getpos(o3chapxdata, x->i<x<i+1)
    dl = diff([map(x->o3chapxdata[x[1],1],posss),i;])
    x = map(x->o3chapxdata[x[1],2],posss)
    ax = reduce(+,map(*,x, dl))/reduce(+,dl)
    o3ls = [o3ls, i+0.5;]
    o3xs = [o3xs, ax;]
end
o3xdata = reshape([o3ls, o3xs;],length(o3ls),2)




#Oxygen, including temperature-dependent Schumann-Runge bands.
o2xdata = readandskip(xsecfolder*"O2.dat",'\t',Float64, skipstart = 3)
function binupO2(list)
    ret = Float64[];
    for i in [176:203;]
        posss = getpos(list[:,1],x->i<x<i+1)
        dl = diff([map(x->list[x[1],1],posss),i;])
        x0 = map(x->list[x[1],2],posss)
        x1 = map(x->list[x[1],3],posss)
        x2 = map(x->list[x[1],4],posss)
        ax0 = reduce(+,map(*,x0, dl))/reduce(+,dl)
        ax1 = reduce(+,map(*,x1, dl))/reduce(+,dl)
        ax2 = reduce(+,map(*,x2, dl))/reduce(+,dl)
        append!(ret,[i+0.5, ax0, ax1, ax2])
    end
    return reshape(ret, 4, 203-176+1).'
end
o2schr130K = readandskip(xsecfolder*"130-190.cf4",'\t',Float64, skipstart = 3)
o2schr130K[:,1] = map(p->1e7/p, o2schr130K[:,1])
o2schr130K = binupO2(o2schr130K)
o2schr190K = readandskip(xsecfolder*"190-280.cf4",'\t',Float64, skipstart = 3)
o2schr190K[:,1] = map(p->1e7/p, o2schr190K[:,1])
o2schr190K = binupO2(o2schr190K)
o2schr280K = readandskip(xsecfolder*"280-500.cf4",'\t',Float64, skipstart = 3)
o2schr280K[:,1] = map(p->1e7/p, o2schr280K[:,1])
o2schr280K = binupO2(o2schr280K)

function o2xsect(T::Float64)
    o2x = deepcopy(o2xdata);
    #fill in the schumann-runge bands according to Minschwaner 1992
    T = clamp(T, 130, 500)
    if 130<=T<190
        o2schr = o2schr130K
    elseif 190<=T<280
        o2schr = o2schr190K
    else
        o2schr = o2schr280K
    end

    del = ((T-100)/10)^2

    for i in [176.5:203.5;]
        posO2 = findfirst(o2x, i)
        posschr = findfirst(o2schr, i)
        o2x[posO2, 2] += 1e-20*(o2schr[posschr, 2]*del^2
                                + o2schr[posschr, 3]*del
                                + o2schr[posschr, 4])
    end

    # add in the herzberg continuum (though tiny)
    # measured by yoshino 1992
    for l in [192.5:244.5;]
        posO2 = findfirst(o2x, l)
        o2x[posO2, 2] += 1e-24*(-2.3837947e4
                            +4.1973085e2*l
                            -2.7640139e0*l^2
                            +8.0723193e-3*l^3
                            -8.8255447e-6*l^4)
    end

    return o2x
end

#HO2
function ho2xsect_l(l::Float64)
    #function to compute HO2 cross-section as a function of wavelength l
    #in nm, as given by Sander 2011 JPL Compilation
    a = 4.91
    b = 30612.0
    sigmamed = 1.64e-18
    vmed = 50260.0
    v = 1e7/l;
    if 190<=l<=250
        return HO2absx = sigmamed / ( 1 - b/v ) * exp( -a * log( (v-b)/(vmed-b) )^2 )
    else
        return 0.0
    end
end
ho2xsect = [190.5:249.5;]
ho2xsect = reshape([ho2xsect, map(ho2xsect_l, ho2xsect);],length(ho2xsect),2)


#H2
h2xdata = readandskip(xsecfolder*"binnedH2.csv",',',Float64, skipstart=4)

#OH
ohxdata = readandskip(xsecfolder*"binnedOH.csv",',',Float64, skipstart=4)
ohO1Dxdata = readandskip(xsecfolder*"binnedOHo1D.csv",',',Float64, skipstart=4)

#SOLAR FLUX
# Change following line as needed depending on local machine
const solarflux=readandskip("/home/mike/Documents/Mars/Photochemistry/Photochemical Data/marssolarphotonflux.dat",'\t',Float64,skipstart=4)[1:2000,:]
solarflux[:,2] = solarflux[:,2]/2


absorber = Dict(:JCO2ion =>:CO2,
                :JCO2toCOpO =>:CO2,
                :JCO2toCOpO1D =>:CO2,
                :JO2toOpO =>:O2,
                :JO2toOpO1D =>:O2,
                :JO3toO2pO =>:O3,
                :JO3toO2pO1D =>:O3,
                :JO3toOpOpO =>:O3,
                :JH2toHpH =>:H2,
                :JOHtoOpH =>:OH,
                :JOHtoO1DpH =>:OH,
                :JHO2toOHpH =>:HO2,
                :JH2OtoHpOH =>:H2O,
                :JH2OtoH2pO1D =>:H2O,
                :JH2OtoHpHpO =>:H2O,
                :JH2O2to2OH =>:H2O2,
                :JH2O2toHO2pH =>:H2O2,
                :JH2O2toH2OpO1D =>:H2O2);

function padtosolar(crosssection::Array{Float64, 2})
    # a function to take an Nx2 array and pad it with zeroes until it's the
    # same length as the solar flux. Returns the cross sections only, as
    # the wavelengths are shared by solarflux

    positions = map(x->findfirst(solarflux[:,1],x),crosssection[:,1])
    retxsec = fill(0.,length(solarflux[:,1]))
    retxsec[positions] = crosssection[:,2]
    retxsec
end

function quantumyield(xsect::Array, arr)
    #function to assemble cross-sections for a given pathway. Inputs are
    #an Nx2 array xsect with wavelengths and photoabsorption cross
    #sections, and arr, a tuple of tuples with a condition and a quantum
    #yield multiplicative factor, either constant or a function of
    #wavelength in the given regime. Return is an array with all of the
    #matching wavelengths and the scaled cross-sections.
    lambdas = Float64[];
    rxs = Float64[];
    for (cond, qeff) in arr
        places = find(cond, xsect[:,1])
        append!(lambdas, xsect[places, 1])
        #if we have a number then map to a function
        isa(qeff, Function) ? (qefffn = qeff) : (qefffn = x->qeff)
        append!(rxs, map(*,map(qefffn, xsect[places, 1]),xsect[places, 2]))
    end

    reshape([lambdas, rxs;],length(lambdas),2)
end

#build the array of cross-sections
crosssection = Dict{Symbol, Array{Array{Float64}}}()
## #this is a dictionary of the 1-nm photodissociation or photoionization
## #cross-sections important in the atmosphere. keys are symbols found in
## #jratelist. each entry is an array of arrays, yielding the wavelengths
## #and cross-sections for each altitude in the atmosphere.
## #
## #NOTE: jspecies refers to the photodissociation or photoionization
## #cross section for a particular species which produces a UNIQUE SET OF
## #PRODUCTS. In this sense, crosssection has already folded in quantum
## #efficiency considerations.

#now add the cross-sections

#CO2 photoionization
setindex!(crosssection,
          fill(co2exdata, length(alt)),
          :JCO2ion)
#CO2+hv->CO+O
setindex!(crosssection,
          map(xs->quantumyield(xs,((l->l>167, 1),
                                   (l->95>l, 0.5))), map(t->co2xsect(t),map(Temp, alt))),
          :JCO2toCOpO)
#CO2+hv->CO+O1D
setindex!(crosssection,
          map(xs->quantumyield(xs,((l->95<l<167, 1),
                                   (l->l<95, 0.5))), map(t->co2xsect(t),map(Temp, alt))),
          :JCO2toCOpO1D)
#O2+hv->O+O
setindex!(crosssection,
          map(xs->quantumyield(xs,((x->x>175, 1),)), map(t->o2xsect(t),map(Temp, alt))),
          :JO2toOpO)
#O2+hv->O+O1D
setindex!(crosssection,
          map(xs->quantumyield(xs,((x->x<175, 1),)), map(t->o2xsect(t),map(Temp, alt))),
          :JO2toOpO1D)

# The quantum yield of O1D from ozone photolysis is actually
# well-studied! This adds some complications for processing.
function O3O1Dquantumyield(lambda, temp)
    if lambda < 306. || lambda > 328.
        return 0.
    end
    temp=clamp(temp, 200, 320)#expression is only valid in this T range

    X = [304.225, 314.957, 310.737];
    w = [5.576, 6.601, 2.187];
    A = [0.8036, 8.9061, 0.1192];
    v = [0.,825.518];
    c = 0.0765;
    R = 0.695;
    q = exp.(-v/(R*temp))
    qrat = q[1]/(q[1]+q[2])

    (q[1]/sum(q)*A[1]*exp.(-((X[1]-lambda)/w[1])^4.)
     +q[2]/sum(q)*A[2]*(temp/300.)^2.*exp.(-((X[2]-lambda)/w[2])^2.)
     +A[3]*(temp/300.)^1.5*exp.(-((X[3]-lambda)/w[3])^2.)
     +c)
end

# O3+hv->O2+O
setindex!(crosssection,
          map(t->quantumyield(o3xdata,
                              (
                               (l->l<193, 1-(1.37e-2*193-2.16)),
                               (l->193<=l<225, l->(1.-(1.37e-2*l-2.16))),
                               (l->225<=l<306, 0.1),
                               (l->306<=l<328, l->(1.-O3O1Dquantumyield(l, t))),
                               (l->328<=l<340, 0.92),
                               (l->340<=l, 1.0)
                               ))
              ,map(Temp, alt)
              ),
          :JO3toO2pO)

# O3+hv->O2+O1D
setindex!(crosssection,
          map(t->quantumyield(o3xdata,
                              (
                               (l->l<193, 1.37e-2*193-2.16),
                               (l->193<=l<225, l->(1.37e-2*l-2.16)),
                               (l->225<=l<306, 0.9),
                               (l->306<=l<328, l->O3O1Dquantumyield(l, t)),
                               (l->328<=l<340, 0.08),
                               (l->340<=l, 0.0)
                               ))
              ,map(Temp, alt)
              ),
          :JO3toO2pO1D)

# O3+hv->O+O+O
setindex!(crosssection,
          fill(quantumyield(o3xdata,((x->true, 0.),)),length(alt)),
          :JO3toOpOpO)

# H2+hv->H+H
setindex!(crosssection,
          fill(h2xdata, length(alt)),
          :JH2toHpH)

# OH+hv->O+H
setindex!(crosssection,
          fill(ohxdata, length(alt)),
          :JOHtoOpH)

# OH+hv->O1D+H
setindex!(crosssection,
          fill(ohO1Dxdata, length(alt)),
          :JOHtoO1DpH)

# HO2+hv->OH+H
setindex!(crosssection,
          fill(ho2xsect, length(alt)),
          :JHO2toOHpH)

# H2O+hv->H+OH
setindex!(crosssection,
          fill(quantumyield(h2oxdata,((x->x<145, 0.89),(x->x>145, 1))),length(alt)),
          :JH2OtoHpOH)
# H2O+hv->H2+O1D
setindex!(crosssection,
          fill(quantumyield(h2oxdata,((x->x<145, 0.11),(x->x>145, 0))),length(alt)),
          :JH2OtoH2pO1D)

# H2O+hv->H+H+O
setindex!(crosssection,
          fill(quantumyield(h2oxdata,((x->true, 0),)),length(alt)),
          :JH2OtoHpHpO)

# H2O2+hv->OH+OH
setindex!(crosssection,
          map(xs->quantumyield(xs,((x->x<230, 0.85),(x->x>230, 1))), map(t->h2o2xsect(t),map(Temp, alt))),
          :JH2O2to2OH)

# H2O2+hv->HO2+H
setindex!(crosssection,
          map(xs->quantumyield(xs,((x->x<230, 0.15),(x->x>230, 0))), map(t->h2o2xsect(t),map(Temp, alt))),
          :JH2O2toHO2pH)

# H2O2+hv->H2O+O1D
setindex!(crosssection,
          map(xs->quantumyield(xs,((x->true, 0),)), map(t->h2o2xsect(t),map(Temp, alt))),
          :JH2O2toH2OpO1D)

lambdas = Float64[]
for j in Jratelist, ialt in 1:length(alt)
    lambdas = union(lambdas, crosssection[j][ialt][:,1])
end

if !(setdiff(solarflux[:,1],lambdas)==[])
    throw("Need a broader range of solar flux values!")
end

#pad all cross-sections to solar
for j in Jratelist, ialt in 1:length(alt)
    crosssection[j][ialt] = padtosolar(crosssection[j][ialt])
end

# we need some global objects for the Jrates calculation:
# intensity as a function of wavelength at each altitude

# this is the unitialized array for storing values
# TODO: do we even need this here?
solarabs = fill(fill(0.,size(solarflux, 1)),length(alt)-2);


function update_Jrates!(n_current::Dict{Symbol, Array{Float64, 1}})
    #this function updates the photolysis rates stored in n_current to
    #reflect the altitude distribution of absorbing species
    #    global solarabs::Array{Array{Float64, 1},1}

    solarabs = Array{Array{Float64}}(length(alt)-2)
    for i in range(1, length(alt)-2)
        solarabs[i] = zeros(Float64, 2000)
    end

    nalt = size(solarabs, 1)
    nlambda = size(solarabs[1],1)

    for jspecies in Jratelist
        species = absorber[jspecies]

        # println(string(jspecies," is absorbed by ",species))
        jcolumn = 0.
        for ialt in [nalt:-1:1;]
            #get the vertical column of the absorbing constituient
            jcolumn += n_current[species][ialt]*dz
            # if jspecies==:JO2toOpO
            #     println(string("At alt = ",alt[ialt+1],
            #                    ", n_",species," = ",n_current[species][ialt],
            #                    ", jcolumn = ",jcolumn))
            #     println("and solarabs[ialt] is $(solarabs[ialt]) before we do axpy")
            #     readline(STDIN)
            # end

            # add the total extinction to solarabs:
            # multiplies air column density at all wavelengths by crosssection
            # to get optical depth
            BLAS.axpy!(nlambda, jcolumn, crosssection[jspecies][ialt+1], 1,
                       solarabs[ialt],1)
        end
    end

    #solarabs now records the total optical depth of the atmosphere at
    #each wavelength and altitude

    # actinic flux at each wavelength is solar flux diminished by total
    # optical depth
    for ialt in [1:nalt;]
        solarabs[ialt] = solarflux[:,2].*exp.(-solarabs[ialt])
    end

    #each species absorbs according to its cross section at each
    #altitude times the actinic flux.
    for j in Jratelist
        for ialt in [1:nalt;]
            n_current[j][ialt] = BLAS.dot(nlambda, solarabs[ialt], 1,
                                          crosssection[j][ialt+1], 1)
        end
    end
end


function timeupdate(mytime)
    for i = 1:10
        update!(n_current, mytime)
    end
    ## plotatm()
    ## show()
    ## yield()
end

## n_current[:H2O]=H2Oinitfrac.*map(z->n_tot(n_current, z),alt[2:end-1])
## [timeupdate(t) for t in [10.0^(1.0*i) for i in -3:14]]
## for i in 1:100
##     update!(n_current, 1e14)
## end
## write_ncurrent(n_current,"converged_standardwater.h5")

## n_current[:H2O]=detachedlayer.*map(z->n_tot(n_current, z),alt[2:end-1])
## [timeupdate(t) for t in [10.0^(1.0*i) for i in -3:14]]
## for i in 1:100
##     update!(n_current, 1e14)
## end
## write_ncurrent(n_current,"converged_highwater.h5")
