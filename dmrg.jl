using LinearAlgebra
using TensorOperations
using KrylovKit

eye(n) = Matrix(1.0I, n, n)

function dosvdtrunc(AA, m)          # AA a matrix; keep at most m states
    F = svd(AA)
    d = F.S
    prob = dot(d, d)                # total probability
    mm = min(m, length(d))          # number of states to keep
    d = d[1:mm]                     # middle matrix in vector form
    trunc = prob - dot(d, d)
    U = F.U[:, 1:mm]
    V = F.Vt[1:mm, :]
    (U, d, V, trunc)                # AA ≈ U * Diagonal(d) * V with error trunc
end

function dosvdleftright(AA, m, toright)
    (U, d, V, trunc) = dosvdtrunc(AA, m)
    if toright
        V = Diagonal(d) * V
    else
        U = U * Diagonal(d)
    end
    (U, V, trunc)
end

function dosvd4(AA, m, toright)     # AA is ia * 2 * 2 * ib; svd down the middle; return two parts
    ia = size(AA, 1)
    ib = size(AA, 4)
    AA = reshape(AA, ia*2, 2*ib)
    (U, V, trunc) = dosvdleftright(AA, m, toright)
    mm = size(U, 2)
    U = reshape(U, ia, 2, mm)
    V = reshape(V, mm, 2, ib)
    (U, V, trunc)
end

function JK(a, b)   # Julia kron, ordered for julia arrays; returns matrix
    (a1, a2) = size(a)
    (b1, b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2], a1*b1, a2*b2)
end

function JK4(a, b)  # Julia kron, ordered for julia arrays, expanded into 4 indices
    (a1, a2) = size(a)
    (b1, b2) = size(b)
    Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2]
end

function dmrg(; n=28, nsweeps=10)   # exact n=28 energy is -12.2254405486
    sz = Float64[0.5 0; 0 -0.5]
    sp = Float64[0 1; 0 0]
    sm = Matrix(sp')
    Htwosite = reshape(JK(sz, sz) + 0.5 * JK(sp, sm) + 0.5 * JK(sm, sp), 2, 2, 2, 2)
    # order for Htwosite is s1, s2, s1p, s2p

    # Initial product state in up down up down ... pattern (Neel state)
    # First tensor is 1 x 2 x m, last is m x 2 x 1 (rather than vectors)
    A = [zeros(1, 2, 1) for _ in 1:n]
    for i in 1:n
        A[i][1, iseven(i) ? 2 : 1, 1] = 1.0
    end

    HLR = [zeros(1, 1) for _ in 1:n]   # initialize to avoid errors on first sweep
    m = 3
    energy = 0.0
    onesite = eye(2)
    for swp in 0:nsweeps
        m = round(Int, 1.3 * m)
        for ii in -n+1:n-1          # if negative, going right to left
            ii == 0 && continue
            i = abs(ii)
            toright = ii > 0

            println("\n sweep, i, dir, m = $swp, $i, ", toright ? "to right" : "to left", " $m")

            dleft = size(A[i], 1)
            alpha = dleft * 2
            dright = size(A[i+1], 3)
            beta = 2 * dright

            HL = zeros(dleft, 2, dleft, 2)
            HR = zeros(2, dright, 2, dright)
            if i > 1
                Aim1 = A[i-1]
                @tensor HL[a,si,ap,sip] := Htwosite[sim1,si,sim1p,sip] * Aim1[b,sim1,a] * Aim1[b,sim1p,ap]
                i > 2 && (HL += JK4(HLR[i-1], onesite))
            end
            HL = reshape(HL, alpha, alpha)
            if i < n-1
                Ai2 = A[i+2]
                @tensor HR[si1,b,si1p,bp] := Htwosite[si1,si2,si1p,si2p] * Ai2[b,si2,a] * Ai2[bp,si2p,a]
                i < n-2 && (HR += JK4(onesite, HLR[i+2]))
            end
            HR = reshape(HR, beta, beta)

            Oleft  = Any[JK(eye(dleft), sz), 0.5*JK(eye(dleft), sp), 0.5*JK(eye(dleft), sm)]
            Oright = Any[JK(sz, eye(dright)), JK(sm, eye(dright)), JK(sp, eye(dright))]

            Ai = A[i]
            Ai1 = A[i+1]
            @tensor AA[a,b,d,e] := Ai[a,b,c] * Ai1[c,d,e]

            # Inefficient implementation: m^4 Ham construction
            Ham = zeros(alpha*beta, alpha*beta)
            for j in 1:length(Oleft)
                Ham += JK(reshape(Oleft[j], alpha, alpha), reshape(Oright[j], beta, beta))
            end
            if i > 1
                Ham += JK(HL, eye(beta))
            end
            if i < n-1
                Ham += JK(eye(alpha), HR)
            end
            bigH = 0.5 * (Ham + Ham')

            vals, vecs, _ = eigsolve(bigH, vec(AA), 1, :SR; ishermitian=true)
            energy = vals[1]
            @show energy
            gr = vecs[1]

            AA = reshape(gr, dleft, 2, 2, dright)

            (A[i], A[i+1], trunc) = dosvd4(AA, m, toright)
            @show trunc
            if toright
                if 1 < i < n-1
                    (i1, i2, i3) = size(A[i])
                    Ai2m = reshape(A[i], i1*i2, i3)
                    @tensor hlri[b,bp] := HL[a,ap] * Ai2m[a,b] * Ai2m[ap,bp]
                    HLR[i] = hlri
                end
            else
                if 1 < i < n-1
                    (i1, i2, i3) = size(A[i+1])
                    Ai12 = reshape(A[i+1], i1, i2*i3)
                    @tensor hlri1[a,ap] := HR[b,bp] * Ai12[a,b] * Ai12[ap,bp]
                    HLR[i+1] = hlri1
                end
            end
        end
    end
    return A, energy
end

A, E = dmrg()
println("\nfinal energy: $E   (exact for n=28: -12.2254405486)")
