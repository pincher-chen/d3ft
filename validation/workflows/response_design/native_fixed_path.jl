# Native DFTK/ForwardDiff path baseline with the registered Miller set fixed.
# The local tagged primal conversion keeps the fixed basis in dual-SCF too.
using DFTK,DftFunctionals,ForwardDiff,JSON3,StaticArrays,LinearAlgebra,Dates,Serialization
struct RegisteredGammaPath end
const RegisteredGammaTag=typeof(ForwardDiff.Tag(RegisteredGammaPath(),Float64))
const REFERENCE_KPOINT=Ref{Any}(nothing)
function fixed_basis!(basis)
    length(basis.kpoints)==1 || error("Registered Gamma-only scope")
    old=basis.kpoints[1];ref=REFERENCE_KPOINT[]
    basis.kpoints[1]=DFTK.Kpoint(old.spin,old.coordinate,copy(ref.G_vectors),copy(ref.mapping),copy(ref.mapping_inv),copy(ref.mapping_device))
    for (i,term) in enumerate(basis.model.term_types);basis.terms[i]=term(basis);end
    DFTK.G_vectors(basis,basis.kpoints[1])==ref.G_vectors || error("Fixed Miller order changed")
    basis
end
function DFTK.construct_value(basis::DFTK.PlaneWaveBasis{T}) where {T<:ForwardDiff.Dual{RegisteredGammaTag}}
    primal=DFTK.PlaneWaveBasis(DFTK.construct_value(basis.model),ForwardDiff.value(basis.Ecut),
        basis.fft_size,basis.variational,basis.kgrid,basis.symmetries_respect_rgrid,
        basis.use_symmetries_for_kpoint_reduction,basis.comm_kpts,basis.architecture)
    fixed_basis!(primal)
end
function main(protocol,structure,outdir)
    p=JSON3.read(read(protocol,String));s=JSON3.read(read(structure,String));ispath(outdir) && error("Immutable output exists");mkpath(outdir)
    function record(name,obj)
        encoded=JSON3.write(obj);temp=joinpath(outdir,name*".tmp");write(temp,encoded*"\n");mv(temp,joinpath(outdir,name*".json");force=true)
    end
    function stage(label;kw...)
        row=(;stage=label,time=string(now()),kw...);record("status",row);println("NATIVE_PATH_STAGE ",JSON3.write(row));flush(stdout)
    end
    try
        DFTK.setup_threading()
        lat0=reduce(hcat,[Float64.(collect(row)) for row in s.cell_angstrom])*Float64(p.angstrom_to_bohr)
        positions=[SVector{3,Float64}(v) for v in s.fractional_positions]
        atoms=[ElementPsp(Symbol(x),load_psp(joinpath(String(p.pp_dir),string(x)*".upf"))) for x in s.symbols]
        theta0=Float64.(collect(p.theta0));direction=Float64.(collect(p.direction));target=Float64(p.target_gamma_eV)
        function basis_at(t;fixed=true)
            theta=theta0+t*direction;T=eltype(theta)
            strain=[theta[1] theta[6]/2 theta[5]/2;theta[6]/2 theta[2] theta[4]/2;theta[5]/2 theta[4]/2 theta[3]]
            model=model_DFT((Matrix{T}(I,3,3)+strain)*lat0,atoms,positions;
                functionals=[DftFunctional(:gga_x_pbe),DftFunctional(:gga_c_pbe)],symmetries=false,temperature=0)
            basis=PlaneWaveBasis(model;Ecut=100,kgrid=(1,1,1),kshift=(0,0,0),fft_size=(144,144,144))
            fixed ? fixed_basis!(basis) : basis
        end
        stage("fixed_basis_preflight")
        b0=basis_at(0.;fixed=false);REFERENCE_KPOINT[]=b0.kpoints[1]
        registered=JSON3.read(read(String(p.native_registered_miller),String))
        Set(Tuple(g) for g in REFERENCE_KPOINT[].G_vectors)==Set(Tuple(Int.(collect(g))) for g in registered) || error("Registered Miller set mismatch")
        checkpoint=deserialize(String(p.native_primal_arrays));psi0=checkpoint.ψ;rho0=checkpoint.ρ
        original=DFTK.energy_hamiltonian(b0,psi0,checkpoint.occupation;ρ=rho0,checkpoint.eigenvalues).ham
        before=(original*psi0)[1];fixed_basis!(b0)
        after=(DFTK.energy_hamiltonian(b0,psi0,checkpoint.occupation;ρ=rho0,checkpoint.eigenvalues).ham*psi0)[1]
        unchanged=maximum(abs.(before-after));unchanged<=1e-10 || error("Basis adapter changes reference Hamiltonian")
        bd=basis_at(ForwardDiff.Dual{RegisteredGammaTag}(0.,1.));bp=DFTK.construct_value(bd)
        DFTK.G_vectors(bp,bp.kpoints[1])==REFERENCE_KPOINT[].G_vectors || error("Dual primal basis lost fixed mask")
        boundary=basis_at(ForwardDiff.Dual{RegisteredGammaTag}(-.03,1.));boundaryp=DFTK.construct_value(boundary)
        DFTK.G_vectors(boundaryp,boundaryp.kpoints[1])==REFERENCE_KPOINT[].G_vectors || error("Endpoint primal basis lost fixed mask")
        record("adapter_checks",(;passed=true,reference_Hamiltonian_max_difference=unchanged,
            reference_plane_waves=length(REFERENCE_KPOINT[].G_vectors),primal_and_dual_fixed_basis_checked=true,
            scope="Local tagged conversion and Kpoint adapter, original native SCF/response algorithms unchanged"))
        # Reuse only the accepted initial native gradient; later proposals use new native dual-SCF.
        accepted=JSON3.read(read(String(p.native_initial_gradient),String));gradient=Float64.(collect(accepted.gap_gradient_eV))
        primal0=JSON3.read(read(String(p.native_primal_json),String));gap=Float64(primal0.gap_eV)
        derivative=dot(gradient,direction);t=0.;psi=psi0;rho=rho0;history=Any[]
        push!(history,(;iteration=0,t,gap_eV=gap,error_eV=gap-target,directional_derivative_eV=derivative,reused_registered_native_base=true))
        record("history",history)
        for iteration in 1:Int(p.design_max_updates)
            move=clamp((target-gap)/derivative,-Float64(p.design_max_step),Float64(p.design_max_step))
            proposal=clamp(t+move,-Float64(p.design_bound),Float64(p.design_bound));abs(proposal-t)>1e-8 || error("Bounded native path stalled")
            stage("native_primal";iteration,t=proposal)
            b=basis_at(proposal);density=rho
            scf=self_consistent_field(b;tol=1e-10,maxiter=150,ψ=psi,ρ=density,
                nbandsalg=DFTK.FixedBands(;n_bands_converge=50,n_bands_compute=54));scf.converged || error("Native primal nonconvergence")
            values=scf.eigenvalues[1];hpsi=(scf.ham*scf.ψ)[1];res=[norm(hpsi[:,j]-values[j]*scf.ψ[1][:,j]) for j=1:50]
            maximum(res)<=1e-7 || error("Native eigensystem gate failed")
            all(abs.(scf.occupation[1][1:48].-2).<1e-6) && maximum(abs.(scf.occupation[1][49:end]))<1e-6 || error("Native occupation gate failed")
            min(values[48]-values[47],values[50]-values[49])>1e-5 || error("Native edge degeneracy")
            trialgap=(values[49]-values[48])*27.211386245988;better=abs(trialgap-target)<abs(gap-target)
            push!(history,(;iteration,t=proposal,gap_eV=trialgap,error_eV=trialgap-target,actual_loss_accepted=better,
                residuals_Ha=res,plane_waves=size(scf.ψ[1],1)));record("history",history)
            better || (stage("completed";converged=false,reason="actual objective did not improve");return)
            t=proposal;gap=trialgap;psi=scf.ψ;rho=scf.ρ
            if abs(gap-target)<=Float64(p.design_tolerance_eV)
                stage("completed";converged=true,updates=iteration,t,gap_eV=gap);return
            end
            iteration==Int(p.design_max_updates) && (stage("completed";converged=false,reason="registered update limit");return)
            stage("native_forward_response";iteration,t)
            dual=self_consistent_field(basis_at(ForwardDiff.Dual{RegisteredGammaTag}(t,1.));tol=1e-10,maxiter=150,ψ=psi,ρ=rho,
                nbandsalg=DFTK.FixedBands(;n_bands_converge=50,n_bands_compute=54))
            dual.converged && all(x.converged for x in dual.response) || error("Native selfconsistent response failure")
            dualgap=(dual.eigenvalues[1][49]-dual.eigenvalues[1][48])*27.211386245988
            derivative=Float64(only(ForwardDiff.partials(dualgap)))
            abs(ForwardDiff.value(dualgap)-gap)<=1e-6 || error("Native primal changed during dual response")
            isfinite(derivative) && abs(derivative)>1e-6 || error("Unresolved native gradient")
            record("response_iter"*string(iteration),(;directional_gap_derivative_eV=derivative,
                response=[(;converged=x.converged,n_iter=x.n_iter,residual_norm=Float64(x.residual_norm)) for x in dual.response]))
        end
    catch err
        stage("failed";error=sprint(showerror,err));rethrow()
    end
end
length(ARGS)==3 || error("protocol structure output required")
main(ARGS[1],ARGS[2],ARGS[3])
