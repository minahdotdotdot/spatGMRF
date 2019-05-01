using LinearAlgebra, SparseArrays, Printf, PyPlot, SpecialFunctions

function genGMRF(a::Float64=5.0)
    stencil = zeros(3,3);
    stencil[1,2]=-1
    stencil[2,1]=-1
    stencil[2,3]=-1
    stencil[3,2]=-1
    stencil[2,2]=a;
    return stencil
end

function genQstencil(a::Float64=5.0, nu::Int=0)
	Q0 = genGMRF(a)
	if nu > 0
		Q = zeros(2*nu+5, 2*nu+5)
		Q[nu+2:nu+4, nu+2:nu+4] = Q0
		for i = 1 : nu
			Q = smoother(Q, stencil=Q0, periodic=false)
		end
		return Q[nu+3:end-1, nu+3:end-1]#[2:end-1, 2:end-1]
	else
		return Q0
	end
end

@inline function no0rem(x::Int, y::Int)
    r = rem(x,y)
    if r == 0
        return y
    else
    	if r > 0
        	return r
        else
        	while r < 0
        		r += y
        	end
        	return r
        end
    end
end

function ijpair(k::Int, m::Int)
	i = no0rem(k,m)
	j = (k-i)/m + 1
	return i, Int(j)
end

function backtok(i::Int, j::Int, m::Int)
	k = m*(j-1)+i
	return k
end

function genQ(m::Int, n::Int, a::Float64=5.0, nu::Int=0; periodic::Bool=true)
	M = m*n
	Qsten = genQstencil(a, nu)
	global Q = spzeros(M, M);
	global IND = zeros(Int,3,4,2);
	global IND[1,:,:]=[0 1; 0 -1; 1 0; -1 0];
	global IND[2,:,:]=[1 1; 1 -1; -1 -1; -1 1];
	global IND[3,:,:]=[0 2; 0 -2; 2 0; -2 0];
	for k =1 : M
		#@printf("%d, \n",k)
		for i = 1 : ceil(Int, nu/2)+1
			for j = i: nu+3-i
				if i==1 && j == 1
					global Q[k,k]=Qsten[i,j]
				else
					if i==1 && j==2
						global ind = IND[1,:,:]
					elseif i==1 && j==3
						global ind = IND[2,:,:]
					elseif i==2 && j==2
						global ind = IND[3,:,:]
					end
					ii, jj = ijpair(k,m)
					#=print([backtok(no0rem(ii+ind[1,1],m),no0rem(jj+ind[1,2],m),m), 
						backtok(no0rem(ii+ind[2,1],m),no0rem(jj+ind[2,2],m),m),
						backtok(no0rem(ii+ind[3,1],m),no0rem(jj+ind[3,2],m),m),
						backtok(no0rem(ii+ind[4,1],m),no0rem(jj+ind[4,2],m),m)],"\n")=#
					X = sparse(
						[k, k, k, k], 
						[backtok(no0rem(ii+ind[1,1],m),no0rem(jj+ind[1,2],m),m), 
						backtok(no0rem(ii+ind[2,1],m),no0rem(jj+ind[2,2],m),m),
						backtok(no0rem(ii+ind[3,1],m),no0rem(jj+ind[3,2],m),m),
						backtok(no0rem(ii+ind[4,1],m),no0rem(jj+ind[4,2],m),m)],
						[Qsten[i,j], Qsten[i,j], Qsten[i,j], Qsten[i,j]],
						M, M);
					global Q = Q + X
				end
			end
		end
	end
	return Q
end

function genQnp(m::Int, n::Int, a::Float64=5.0, nu::Int=0; periodic::Bool=true)
	M = m*n
	Qsten = genQstencil(a, nu)
	if nu ==0
		global Q = (spzeros(M, M) + a*I 
			+sparse(1:M-1, 2:M,-1*ones(M-1),M,M)
			+sparse(2:M, 1:M-1,-1*ones(M-1),M,M)
		)
		return Q
	else
		for i = 1 : ceil(Int, nu/2)+1
			for j = i:nu+3-i
				#@printf("%d, %d, %4.2f \n", i, j, Qsten[i,j])
				if i ==1 && j ==1
					global Q = Qsten[1,1]*I
				else
					X = (sparse(1:M-m*(i-1)-(j-1), 1+m*(i-1)+(j-1):M,
							Qsten[j,i]*ones(M-m*(i-1)-(j-1)),M,M
							) #sup
						+ sparse(1+m*(j-1)+(i-1):M, 1:M-m*(j-1)-(i-1),  
							Qsten[j,i]*ones(M-m*(j-1)-(i-1)),M,M
							) 
						)#sub
					if i == j
						global Q = Q + X
					else
						global Q = Q + X + X'
					end
				end
			end
		end
	end
	return Q
end
function smoother(q::Array{T,2}; 
    stencil::Array{T,2}=genStencil(5,5),
    periodic::Bool=true) where T<:Real
    m, n = size(stencil);
    mm = floor(Int,m/2)
    nn = floor(Int,n/2)
    grid_y, grid_x = size(q)
    smoothed = Array{Float64,2}(undef, grid_y, grid_x)
    if periodic==false
        for j = 1 : grid_y
            for i = 1 : grid_x
                top_s, bottom_s, left_s, right_s = 1,m,1,n;
                top_q, bottom_q, left_q, right_q = j-mm,j+mm,i-nn,i+nn;
                if j<=mm
                    top_s=mm-j+2;           top_q=1;
                elseif j>grid_y-mm
                    bottom_s=mm+grid_y-j+1; bottom_q=grid_y;
                end
                if i<=nn
                    left_s=nn-i+2;          left_q=1;
                elseif i>grid_x-nn
                    right_s=nn+grid_x-i+1;  right_q=grid_x;
                end
                smoothed[j,i] = 1/sum(stencil[top_s:bottom_s, left_s:right_s])*(
                    sum(q[top_q:bottom_q, left_q:right_q]
                        .*stencil[top_s:bottom_s, left_s:right_s])
                    )
            end
        end
        return smoothed
    else
        q = hcat(q[:,end-nn+1:end], q, q[:, 1:nn])
        for j = 1 : grid_y
            for i = nn+1 : grid_x+nn
                top_s, bottom_s, left_s, right_s = 1,m,1,n;
                top_q, bottom_q, left_q, right_q = j-mm,j+mm,i-nn,i+nn;
                if j<=mm
                    top_s=mm-j+2;           top_q=1
                elseif j>grid_y-mm
                    bottom_s=mm+grid_y-j+1; bottom_q=grid_y
                end
                smoothed[j,i-nn] = 1/sum(stencil[top_s:bottom_s, left_s:right_s])*(
                    sum(q[top_q:bottom_q, left_q:right_q]
                        .*stencil[top_s:bottom_s, left_s:right_s])
                    )
            end
        end
        return smoothed
    end
end

function showQ(m::Int, n::Int, a::Float64=5.0, nu::Int=0)
	Q = Array(genQ(m, n, a, nu))
	subplot(121)
	colorbar(imshow(Q, cmap="viridis"))
	subplot(122)
	colorbar(imshow(inv(Q), cmap="viridis"))
end
```
Function that performs numIters iterations of gauss-seidel on 
Ax = b with x as the initial guess.  Each iteration is 
accomplished by solving the usual (D-L)x = Ux + b by 

Step 1: Form y = Ux + b via column-oriented matvec 
Step 2: Solve (D-L)x = y via column-oriented forward sub 
```
function solgaussseidel!(A::SparseMatrixCSC, x::Vector, b::Vector,
	numIters::Int64, omega::Float64=4.0/5.0;tol::Float64=0.0)

	numRows,numCols = size(A); 
	
	if (numRows != numCols) 
		error("matrix is not square")
	end 

	if (numRows != size(b,1))
		error("matrix and rhs vector of inconsistent size")
	end 

	if (numCols != size(x,1))
		error("matrix and initial guess of inconsistent size")
	end 
	if tol==0.0
		for kk=1:numIters 

			dElem = 0.0;
			for jj in 1:size(A,1);  
				xhat = b[jj];
				for ii = A.colptr[jj]:A.colptr[jj+1]-1
					row = A.rowval[ii]; 
					val = A.nzval[ii]; 
					if (row == jj) 
						dElem = val; 
					else 
						xhat -= val*x[row]; 
					end 
				end 
				x[jj] = xhat/dElem; 
			end 

		end 
	else
		while norm(Q*x-y)>tol

			dElem = 0.0;
			for jj in 1:size(A,1);  
				xhat = b[jj];
				for ii = A.colptr[jj]:A.colptr[jj+1]-1
					row = A.rowval[ii]; 
					val = A.nzval[ii]; 
					if (row == jj) 
						dElem = val; 
					else 
						xhat -= val*x[row]; 
					end 
				end 
				x[jj] = xhat/dElem; 
			end 

		end
	end
	
end 

function genGrd(m::Int, n::Int)
	grd=zeros(Int, m*n,2);
	count = 0
	while count <m*n
		for j = 1 : n
			for i = 1 : m
				count +=1
				grd[count,1]=Float64(i);
				grd[count,2]=Float64(j);
			end
		end
	end
	return grd
end

function genRdist(grd::Array)
	M = size(grd)[1]
	rdist = zeros(M,1)
	for i = 1 : M
		rdist[i,1] = norm(grd[i,:]-grd[1,:])
	end
	return rdist
end

function matern(x::Float64, nu::Int, a::Float64)
	kappa = sqrt(a-4)
	coeff=1
	return coeff*(kappa*x)^(nu)*besselk(nu, kappa*x)
end
