using LinearAlgebra
using TensorOperations

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

function tebd(; n=28, m=10, nsweeps=10000)
    sz = Float64[0.5 0; 0 -0.5]
    sp = Float64[0 1; 0 0]
    sm = Matrix(sp')

    Htwosite = Float64[sz[s1,s1p] * sz[s2,s2p] + 0.5 * (sp[s1,s1p] * sm[s2,s2p] + sm[s1,s1p] * sp[s2,s2p])
                       for s1=1:2, s2=1:2, s1p=1:2, s2p=1:2]

    # Initial product state in up down up down ... pattern (Neel state)
    # First tensor is 1 x 2 x m, last is m x 2 x 1 (rather than vectors)
    A = [zeros(1, 2, 1) for _ in 1:n]
    for i in 1:n
        A[i][1, iseven(i) ? 2 : 1, 1] = 1.0
    end

    tau = 0.2
    totE = 0.0
    for swp in 1:nsweeps
        swp % 100 == 0 && (tau = 0.2 * 100 / swp)
        taugate = reshape(exp(-tau * reshape(Htwosite, 4, 4)), 2, 2, 2, 2)
        totE = 0.0
        for ii in -n+1:n-1          # if negative, going right to left
            ii == 0 && continue
            i = abs(ii)
            toright = ii > 0

            Ai = A[i]
            Ai1 = A[i+1]
            @tensor AA[a,f,g,e] := Ai[a,b,c] * Ai1[c,d,e] * taugate[b,d,f,g]
            @tensor HAA[a,f,g,e] := Htwosite[f,g,fp,gp] * AA[a,fp,gp,e]
            nor = dot(AA, AA)
            SdotS = dot(AA, HAA)

            toright && (totE += SdotS / nor)
            AA *= 1.0 / sqrt(nor)
            (A[i], A[i+1], _) = dosvd4(AA, m, toright)
        end
        swp % 10 == 0 && println("$swp $tau $totE")
    end
    return A, totE
end

tebd()
