using TensorOperations

# Contract a random network with itself using only @tensor.
function contract_selfnetwork(; m=20, p=2)   # p = physical dim
    D = p
    N = [rand(D, D, p) for _ in 1:m]
    N[1] = rand(1, D, p)   # rand(D,p) does not work: keep a trivial leg
    N[m] = rand(D, 1, p)

    local Result
    for i in 1:m
        Ni = N[i]
        if i == 1
            @tensor New_Result[a,b,c,d] := Ni[a,b,e] * Ni[c,d,e]
        else
            @tensor New_Result[a,f,c,g] := Result[a,b,c,d] * Ni[b,f,e] * Ni[d,g,e]
        end
        Result = New_Result
    end
    return Result
end

R = contract_selfnetwork()
println("size: ", size(R), "  value: ", R[1])
