%% Iterative Soft-thresholding Algo. for sparse K prediction
% By Myung (Michael) Cho
%
% ----

function Finit = ISTA_feasible_init(A,B1,B2,Q,R,s,gam,options)

%% Parameters
rhoinit  = options.rho; % step size
blksize = options.blksize;

%% size of the input matrix
[n,m] = size(B2);

%% absolute tolerances for the stopping criterion
Max_Iter = 1000;
rho=rhoinit;

F = lqr(A,B2,Q,R);


%% define function
max_eig = @(F) max(real(eig(A-B2*F)));
lyap_P = @(F) lyap((A - B2 * F)', Q + F' * R * F);
lyap_L = @(F) lyap((A - B2 * F), B1*B1');
gradJ_fun = @(F,P,L) 2*(R*F - B2'*P)*L;
% shrinkage operator
switch options.method
    case 'l1'
        % shrinkage operator
        Gval = @(F) norm(F(:),1);
        Fobj = @(F) trace(B1' * lyap_P(F) * B1) + gam*norm(F(:),1);
    case 'blkl1'
        % block shrinkage operator
        Gval = @(F) Gobj_blkl1(F,blksize)
        Fobj = @(F) trace(B1' * lyap_P(F) * B1) + gam*Gobj_blkl1(F,blksize);
end

%% Solve the minimization problem using Iterative Soft-Thresholding
for itr=1:Max_Iter
    stable_flag=1;
    F_pre = F;
    
    P = lyap_P(F_pre);
    L = lyap_L(F_pre);
    
    gradJ = gradJ_fun(F_pre, P, L);
    
    while stable_flag == 1
        F = F_pre - gradJ/rho; % gradient descent
        switch options.method
            case 'l1'
                % shrinkage operator
                F = shrinkage(F,ones(size(F)),gam,rho);
            case 'wl1'
                % shrinkage operator
                F = shrinkage(F,W,gam,rho);
            case 'blkl1'
                % block shrinkage operator
                sub_mat_size = size(F)./blksize;
                F = block_shrinkage(F,...
                    ones(size(F)./blksize),gam,rho,blksize,sub_mat_size);
            case 'blkwl1'
                % block shrinkage operator
                sub_mat_size = size(F)./blksize;
                F = block_shrinkage(F,W,gam,rho,blksize,sub_mat_size);
        end
        
        
        %% stability check
        alpha=0.0001;
        beta=1.5;
        if (max_eig(F) >= 0 || Fobj(F) > Fobj(F_pre) - alpha/rho*norm(gradJ)^2 )
            % backtacking with small step size
            rho = rho*beta;
            stable_flag=1;
            if rho > 10^16
                disp('small step size - break');
                break;
            end
        else
            rho = rhoinit;
            stable_flag=0;
        end
    end
    
    if Gval(F) <= s
        Finit = F;
        break;
    end
end


if (itr == Max_Iter) && (~quiet)
    disp('Maximum number of ISTA steps reached!')
    disp('Operation Fail');
end


end