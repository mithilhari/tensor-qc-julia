# Julia 1.x ports

Install dependencies once:

    julia -e 'using Pkg; Pkg.add(["TensorOperations", "KrylovKit"])'

Run any file with `julia <file>.jl`.

| File | What changed |
|---|---|
| TensorContract.jl | `if` moved outside `@tensor`; wrapped in a function; prints matrix vs tensor route agreement |
| j.jl | same `@tensor`/`if` fix; wrapped in a function |
| tebdshort.jl | `eye`→removed (unused), `expm`→`exp`, `diagm`→`Diagonal`, `scalar(...)` inside `@tensor`→`dot`, `using LinearAlgebra`, wrapped in `tebd()` |
| htwo.jl | `expm`→`exp`, `diagm`→`Diagonal`, `diag` renamed (it shadowed `LinearAlgebra.diag`), `D` passed in instead of a global, wrapped in a function |
| dmrg_test.jl | functions defined before use; undefined `dV` fixed; SVD now returns V' (was missing the transpose); index order in the pair contraction fixed so reshape matches; isometry checks added |
| dmrg.jl | `eye` helper, `eigs`→`KrylovKit.eigsolve`, `diagm`→`Diagonal`, wrapped in `dmrg()`; prints final energy vs exact |
