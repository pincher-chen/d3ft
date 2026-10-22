# DFTK 0.7.16 common-input forward gate, not a derivative performance claim.
using DFTK
using DftFunctionals
using JSON3
using StaticArrays
using Unitful
using UnitfulAtomic
using Dates

length(ARGS) in (5,6,7) || error("structure.json pp_dir output_dir cutoff grid [angstrom_to_bohr [k]]")
structure_file, pp_dir, output_dir, cutoff_str, grid_str = ARGS[1:5]
ispath(output_dir) && error("immutable output directory already exists")
mkpath(output_dir)
function stage(name; extra...)
    open(joinpath(output_dir, "status.json"), "w") do io
        JSON3.write(io, (; stage=name, time=string(now()), job=get(ENV,"SLURM_JOB_ID",""),
                          step=get(ENV,"SLURM_STEP_ID",""), extra...))
    end
    println("response_design_DFTK_STAGE ", name)
    flush(stdout)
end

try
    stage("initializing")
    spec = JSON3.read(read(structure_file, String))
    # D3FT/ASE stores lattice vectors as rows; DFTK stores them as columns.
    lattice = reduce(hcat, [Float64.(collect(row)) for row in spec.cell_angstrom])
    # Optional explicit conversion matches D3FT's actual imported legacy
    # geometry in Bohr. Native published runs retain their original units.
    angstrom_to_bohr=length(ARGS)>=6 ? parse(Float64,ARGS[6]) : austrip(1u"angstrom")
    lattice *= angstrom_to_bohr
    positions = [SVector{3,Float64}(p) for p in spec.fractional_positions]
    atoms = [ElementPsp(Symbol(s), load_psp(joinpath(pp_dir, string(s)*".upf")))
             for s in spec.symbols]
    functionals = [DftFunctional(:gga_x_pbe), DftFunctional(:gga_c_pbe)]
    model = model_DFT(lattice, atoms, positions; functionals, symmetries=false, temperature=0)
    grid = parse(Int, grid_str)
    nk=length(ARGS)==7 ? parse(Int,ARGS[7]) : 2
    basis = PlaneWaveBasis(model; Ecut=parse(Float64,cutoff_str),
                           kgrid=(nk,nk,nk), kshift=(0,0,0), fft_size=(grid,grid,grid))
    open(joinpath(output_dir,"metadata.json"),"w") do io
        JSON3.write(io, (; julia=string(VERSION), dftk=string(pkgversion(DFTK)),
            structure_file, pp_dir, Ecut=basis.Ecut, fft_size=basis.fft_size,
            kgrid=[nk,nk,nk], kshift=[0,0,0], symmetry=false,
            n_electrons=model.n_electrons, angstrom_to_bohr,
            basis_convention="per-k |G+k| kinetic sphere",
            plane_waves_per_k=[length(kp.G_vectors) for kp in basis.kpoints],
            k_fractional=[collect(kp.coordinate) for kp in basis.kpoints],
            unit="Hartree/Bohr", pilot_only=true))
    end
    stage("self_consistent_field")
    start=time()
    scfres=self_consistent_field(basis; tol=1e-10, maxiter=150)
    solve_seconds=time()-start
    stage("forces_and_stress")
    f=compute_forces_cart(scfres)
    stress=compute_stresses_cart(scfres)
    stress_rows=[collect(stress[i,:]) for i in 1:3]
    pressure=-sum(stress[i,i] for i in 1:3)/3*29421.015697
    results=(; energy_Ha=scfres.energies.total, pressure_GPa=pressure,
        stress_Ha_Bohr3=stress_rows, forces_Ha_Bohr=[collect(x) for x in f],
        converged=scfres.converged, iterations=scfres.n_iter,
        residual_norm=hasproperty(scfres,:norm_Δρ) ? scfres.norm_Δρ : nothing,
        solve_seconds, total_seconds=time()-start, pilot_only=true,
        structural_relaxation_performed=false)
    open(joinpath(output_dir,"point.json"),"w") do io
        JSON3.write(io,results)
    end
    println("response_design_DFTK_POINT ",JSON3.write(results))
    scfres.converged || error("DFTK electronic convergence gate failed")
    stage("completed_forward_gate")
catch err
    stage("failed"; error=sprint(showerror,err))
    rethrow()
end
