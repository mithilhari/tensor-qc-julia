using TensorOperations

# Contract a random MPS-like network with itself two ways and compare:
#   1. pure tensor contraction (M)
#   2. reshape + matrix multiplication (Result)
function contract_network(; m=20, D=10, p=2)   # m = network size, D = bond dim, p = physical dim
    N = [rand(D, D, p) for _ in 1:m]   # random tensor network
    N[1] = rand(1, D, p)               # first and last tensors have a trivial outer leg
    N[m] = rand(D, 1, p)

    local Result, M
    for i in 1:m
        Ni = N[i]
        @tensor A[a,c,b,d] := Ni[a,b,e] * Ni[c,d,e]   # contract each tensor with itself

        # Tensor contraction only
        if i == 1
            @tensor O[a,b,c,d] := Ni[a,b,e] * Ni[c,d,e]
        else
            @tensor O[a,f,c,g] := M[a,b,c,d] * Ni[b,f,e] * Ni[d,g,e]
        end
        M = O

        # Reshape tensors and matrix multiplication
        if i == 1
            Result = reshape(A, 1, D*D)
        elseif i == m
            Result = Result * reshape(A, D*D, 1)
        else
            Result = Result * reshape(A, D*D, D*D)
        end
    end
    return Result[1], M[1]
end

r_mat, r_tensor = contract_network()
println("matrix route:  ", r_mat)
println("tensor route:  ", r_tensor)
println("relative diff: ", abs(r_mat - r_tensor) / abs(r_tensor))
