using LinearAlgebra
using TensorOperations

function JK(a, b)   # Julia kron, ordered for julia arrays; returns matrix
    (a1, a2) = size(a)
    (b1, b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2], a1*b1, a2*b2)
end

function dosvdleftright(A, toright, D)   # keep at most D states, normalize singular values
    F = svd(A)
    U, s, Vt = F.U, F.S, F.Vt
    if length(s) > D
        s = s[1:D]
        U = U[:, 1:D]
        Vt = Vt[1:D, :]
    end
    s = s / norm(s)
    if toright
        Vt = Diagonal(s) * Vt
    else
        U = U * Diagonal(s)
    end
    (U, s, Vt)
end

function run_htwo(; p=2, m=20, D=10, nsweeps=1000)   # p = physical dim
    N = [zeros(1, 2, 1) for _ in 1:m]
    for i in 1:m
        N[i][1, iseven(i) ? 1 : 2, 1] = 1.0
    end

    sz = Float64[0.5 0; 0 -0.5]
    sp = Float64[0 1; 0 0]
    sm = Matrix(sp')
    Htwosite = JK(sz, sz) + 0.5 * JK(sp, sm) + 0.5 * JK(sm, sp)
    tau = 0.2
    gate = reshape(exp(-Htwosite * tau), 2, 2, 2, 2)

    for swp in 1:nsweeps
        if swp % 100 == 0
            tau = 2 / swp
            gate = reshape(exp(-Htwosite * tau), 2, 2, 2, 2)
        end
        for k in (-m+1):(m-1)
            k == 0 && continue
            toright = k > 0
            j = abs(k)
            Nj = N[j]
            Nj1 = N[j+1]
            @tensor M[a,f,g,d] := gate[b,e,f,g] * Nj[a,b,c] * Nj1[c,e,d]
            s = size(M)
            O = reshape(M, s[1]*s[2], s[3]*s[4])
            (u, sv, v) = dosvdleftright(O, toright, D)

            dim = length(sv)
            N[j] = reshape(u, s[1], p, dim)     # store the changes
            N[j+1] = reshape(v, dim, p, s[4])
        end
    end
    return N
end

N = run_htwo()
println("done; bond dims: ", [size(t, 3) for t in N[1:end-1]])
