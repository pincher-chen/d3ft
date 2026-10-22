# Regression for the actual nested ForwardDiff ordering failure in DFTK Ewald.
using ForwardDiff
struct RegisteredGammaPath end
const RegisteredGammaTag=typeof(ForwardDiff.Tag(RegisteredGammaPath(),Float64))
f(x)=ForwardDiff.derivative(y -> (x+y)^3, x)
old_failed=false
try
    f(ForwardDiff.Dual{RegisteredGammaPath}(.2,1.))
catch err
    global old_failed=err isa ForwardDiff.DualMismatchError
end
@assert old_failed "Original raw tag should reproduce nested-tag failure"
result=f(ForwardDiff.Dual{RegisteredGammaTag}(.2,1.))
@assert isapprox(ForwardDiff.value(result),.48;atol=1e-14)
@assert isapprox(only(ForwardDiff.partials(result)),4.8;atol=1e-14)
println("PASS: original tag failure reproduced; standard Tag preserves nested first/second derivatives")
