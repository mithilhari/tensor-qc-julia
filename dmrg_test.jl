using LinearAlgebra
using TensorOperations

# Bring a random MPS into mixed-canonical form around site `index`:
# sites left of it become left-isometries, sites right of it right-isometries.

function JK(a, b)   # Julia kron, ordered for julia arrays; returns matrix
    (a1, a2) = size(a)
    (b1, b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2], a1*b1, a2*b2)
end

# SVD; singular values go into V (toright) or U (to left). Optionally normalize them.
function dosvdleftright(A, toright; normalize=false)
    F = svd(A)
    U, s, Vt = F.U, F.S, F.Vt
    normalize && (s = s / norm(s))
    if toright
        Vt = Diagonal(s) * Vt
    else
        U = U * Diagonal(s)
    end
    (U, s, Vt)
end

function canonicalize(; p=2, m=20, D=10, index=4)   # p = physical dim; index must be in 3:m-2
    # Tensor layout: (left bond, physical, right bond)
    N = [rand(D, p, D) for _ in 1:m]
    N[1] = rand(1, p, D)
    N[m] = rand(D, p, 1)

    # Left sweep: sites 1 .. index-2 become left-isometric
    for i in 1:index-2
        Ni = N[i]; Ni1 = N[i+1]
        @tensor M[a,b,e,d] := Ni[a,b,c] * Ni1[c,e,d]     # (left, phys_i, phys_i+1, right)
        s = size(M)
        O = reshape(M, s[1]*s[2], s[3]*s[4])
        (u, sv, v) = dosvdleftright(O, true; normalize = (i+1 == index-1))
        dim = length(sv)
        N[i]   = reshape(u, s[1], s[2], dim)
        N[i+1] = reshape(v, dim, s[3], s[4])
    end

    # Right sweep: sites m .. index+2 become right-isometric
    for j in m:-1:index+2
        Nj1 = N[j-1]; Nj = N[j]
        @tensor M[a,b,e,d] := Nj1[a,b,c] * Nj[c,e,d]
        s = size(M)
        O = reshape(M, s[1]*s[2], s[3]*s[4])
        (u, sv, v) = dosvdleftright(O, false; normalize = (j-1 == index+1))
        dim = length(sv)
        N[j-1] = reshape(u, s[1], s[2], dim)
        N[j]   = reshape(v, dim, s[3], s[4])
    end
    return N
end

index = 4
N = canonicalize(index=index)

# Check isometry conditions
for i in 1:index-2
    a, b, c = size(N[i]); L = reshape(N[i], a*b, c)
    println("site $i left-isometric:  ", L' * L ≈ I)
end
for j in index+2:length(N)
    a, b, c = size(N[j]); R = reshape(N[j], a, b*c)
    println("site $j right-isometric: ", R * R' ≈ I)
end

sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = Matrix(sp')
Htwosite = reshape(JK(sz, sz) + 0.5 * JK(sp, sm) + 0.5 * JK(sm, sp), 2, 2, 2, 2)
