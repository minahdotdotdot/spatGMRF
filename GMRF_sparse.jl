include("GMRF_functions.jl")

#=
Parameters and domain size
Always have square domains on unit grid (m,n)
nu, a, rho are smoothness and range parameters for the Matern cov func.
=#

m=150;n=m; M = m*n
nu = 1;
a = 4.5
rho = sqrt(8)/sqrt(a-4)

#=
Generate a periodic precision matrix based on GMRF stencil.
=#
Q = genQ(m,n,a,nu);
@printf("Done generating Q.\n")

#=
Exploit sparse structure of Q:
Use Gauss-seidel to approximate the first column of Q^{-1}=Sigma.
Use periodic domain (and resulting circulant structure of Q and Sigma),
to approximate Sigma with just Sigma[1,:]. 
Sigma should also be symmetric. 
=#
x=zeros(Q.m);y=zeros(Q.m);y[1]=1.0;
solgaussseidel!(Q,x,y,10,tol=1e-12); 
print(norm(Q*x-y))
print("\n")
x = vcat(x[end:-1:2], x)

#=
Scale down to Correlation (from Covariance).
=#
Cor1=x[M:end]/x[M];

#=
Assuming Q was positive semi-definite, 
plot correlations derived from GMRF Q 
vs.
the exact Matern correlation it's trying to approximate. 
=#
grd = genGrd(m,n);
d =genRdist(grd); d[1] = 1e-17;
V = hcat(d, Cor1);
p = sortperm(V[:,1])
V = V[p,:]
VV = unique(V[:,1])
SS = zeros(length(VV))
dM=1
while dM <= length(VV)
	VVV = V[(V[:,1].==VV[dM]),2]
	SS[dM]=sum(VVV)/length(VVV)
	if VV[dM]>10
		break
	end
	global dM +=1
end
scatter(VV[1:dM-1],SS[1:dM-1],c="g")
xx = range(1e-17, stop=10, length=1000);
Matern2=matern.(xx, nu, a); Matern2=Matern2/Matern2[1];
plot(xx,Matern2,c="r")
axhline(c="k"); axvline(c="k");
title("Range parameter= "*string(rho)*", m="*string(m))
