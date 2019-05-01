include("GMRF_functions.jl")
#=
Parameters and domain size
Always have square domains on unit grid (m,n)
nu, a, rho are smoothness and range parameters for the Matern cov func.
=#
m=7;n=m; M = m*n
nu = 1;
a = 4.5
rho = sqrt(8)/sqrt(abs(a)-4)
@printf("The range parameter is %f.\n", rho)
#=
Generate a periodic precision matrix based on GMRF stencil.
=#
Q = genQ(m,n,a,nu);
@printf("Done generating Q.\n")
#=
fig, ax = subplots(); 
fig.colorbar(ax.imshow(Q))
savefig("Q_rho="*string(rho)*", m="*string(m)*"nu="*string(nu)*".png", 
	pad_inches=.10, bbox_inches="tight")
close(fig)
=#

#=
Just compute inv(Q)
=#

Sigma=inv(Array(Q));

#=
Scale down to Correlation (from Covariance).
Now we can use the circulant property of Sigma, and just use the first row. 
=#
Cor1=Sigma[1,1:end]/Sigma[1,1]
Cor = zeros(Q.m,Q.n)
for j = 1 : m*n
	global Cor[j,:]=Cor1[no0rem.(1+(j-1):M+(j-1), M)]
end

#=
Build the whole correlation matrix just to look at it.
=#
fig, ax = subplots(); 
fig.colorbar(ax.imshow(Cor))
savefig("Cor_rho="*string(rho)*", m="*string(m)*"nu="*string(nu)*".png", 
	pad_inches=.10, bbox_inches="tight")
close(fig)
#=
#=
Check if Q is positive semi-definite.
If it is, 
plot correlations derived from GMRF Q 
vs.
the exact Matern correlation it's trying to approximate. 
=#
if minimum(eigvals(Array(Q)))>0
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
	#dM = ceil(Int,length(VV)/40)
	scatter(VV[1:dM-1],SS[1:dM-1],c="g")
	xx = range(1e-17, stop=10, length=1000);
	Matern2=matern.(xx, nu, a); Matern2=Matern2/Matern2[1];
	plot(xx,Matern2,c="r")
	axhline(c="k"); axvline(c="k");
	title("Range parameter= "*string(rho)*", m="*string(m))
else
	@printf("Q is not positive semi-definite!\n")
end
=#