# Independent DFTK radial form factors; does not run SCF or change PP data.
using DFTK, JSON3
length(ARGS)==2 || error("pp_dir immutable_output.json")
ppdir,output=ARGS
ispath(output) && error("immutable output already exists")
q=collect(range(0.001,12.0,length=129))
results=Dict()
for symbol in ["Mg","Si","N","Ge"]
    p=load_psp(joinpath(ppdir,symbol*".upf"))
    channels=[]
    for l in 0:p.lmax
        n=size(p.h[l+1],1)
        push!(channels,Dict("l"=>l,"h"=>[collect(p.h[l+1][i,:]) for i in 1:n],
            "beta_fourier"=>[[DFTK.eval_psp_projector_fourier(p,i,l,x) for x in q] for i in 1:n]))
    end
    results[symbol]=Dict("valence"=>p.Zion,"channels"=>channels,
        "local_fourier"=>[DFTK.eval_psp_local_fourier(p,x) for x in q],
        "alpha"=>DFTK.eval_psp_energy_correction(Float64,p,1))
end
open(output,"w") do f
    JSON3.write(f,Dict("q_Bohr_inverse"=>q,"elements"=>results,
        "DFTK"=>string(pkgversion(DFTK)),"Julia"=>string(VERSION)))
end
println("response_design_PP_FORM_FACTORS_READY")
