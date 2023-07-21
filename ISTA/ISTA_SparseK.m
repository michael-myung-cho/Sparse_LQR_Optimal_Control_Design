%% Iterative Soft-thresholding Algo. for sparse K prediction
% By Myung (Michael) Cho
%
% ----

function solpath = ISTA_SparseK(A,B1,B2,Q,R,options)

%% Parameters
gamval = options.gamval; % sparsity level
rhoinit  = options.rho; % step size
blksize = options.blksize;


%% size of the input matrix
[n,m] = size(B2);

%% preallocate memory for output variables
solpath.F    = zeros(m,n,length(gamval));
solpath.nnz  = zeros(1,length(gamval));
solpath.J    = zeros(1,length(gamval));
solpath.gam  = zeros(1,length(gamval));
solpath.Fopt = zeros(m,n,length(gamval));
solpath.Jopt = zeros(1,length(gamval));
solpath.tBuf = zeros(1,length(gamval));
solpath.itrBuf = zeros(1,length(gamval));

%% set the number of reweighted scheme for
% the weighted l1 or the weighted sum of Frobenius norms
if strcmp(options.method, 'wl1') || strcmp(options.method, 'blkwl1')
    reweighted_Max_Iter = options.reweightedIter;
else
    reweighted_Max_Iter = 1;
end

% weight matrix for weighted l1
if strcmp(options.method, 'wl1')
    W = ones(m,n);
end

% weight matrix for sum of Frobenius norm
% sub_mat_size is numbers of rows and columns of partitioned block submatrices
if strcmp(options.method, 'blkwl1')
    sub_mat_size = [m,n]./blksize;
    W = ones(sub_mat_size);
end


%% absolute tolerances for the stopping criterion
eps_abs = 1.e-4;
ISTA_Max_Iter = options.maxiter;


%% display result
quiet = 1; % display option

%% LQR point
F = lqr(A,B2,Q,R);


%% define funtion

max_eig = @(F) max(real(eig(A-B2*F)));
lyap_P = @(F) lyap((A - B2 * F)', Q + F' * R * F);
lyap_L = @(F) lyap((A - B2 * F), B1*B1');
gradJ_fun = @(F,P,L) 2*(R*F - B2'*P)*L;
Jobj = @(F) trace(B1' * lyap_P(F) * B1);
Jobj_UB = @(F,F_pre,gradJ,rho) Jobj(F_pre) + trace((F - F_pre)'*gradJ) + rho/2*norm(F-F_pre,'fro')^2;


%% Simulation starts
for k = 1:length(gamval)
    tITAs=tic;
    gam = gamval(k);
    rho = rhoinit;
    
    for reweightedstep = 1 : reweighted_Max_Iter
        
        %% Solve the minimization problem using Iterative Soft-Thresholding
        for itr=1:ISTA_Max_Iter
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
                beta=1.5;
                if (max_eig(F) >= 0 || Jobj(F) > Jobj_UB(F,F_pre,gradJ,rho) )
                    % backtacking with small step size
                    rho = rho*beta;
                    stable_flag=1;
                    if rho > 10^5
                        disp('small step size - break');
                        break;
                    end
                else
                    %fprintf('innItr:%d, \t, rho:%f, \t gradJ mag: %f, \t Fobj-Fobj_pre: %f \n ', innItr, rho, norm(gradJ/rho,'fro'),norm(Fobj(F)-Fobj(F_pre),'fro'));
                    rho = rhoinit;
                    stable_flag=0;
                end
            end
            
            % stoppin criterion
            resF = norm(F-F_pre,'fro');
            if  (resF < eps_abs && itr > 10 && stable_flag == 0) % converge stop
                %disp('converge stop');
                break;
            end
            %% display
            if ~quiet && stable_flag == 0
                fprintf('%5d \t %6.1f \t %6.3f \t %6.3f \t %6.3f \n', itr, gam, resF, norm(gradJ,'fro'), Fobj(F,gam));
            end
            
        end
        
        if (itr == ISTA_Max_Iter) && (~quiet)
            disp('Maximum number of ISTA steps reached!')
            disp(['The residual is ', num2str(resF,'%10.4E')]);
        end
        
        if max(real(eig(A-B2*F))) < 0
            % do nothing
        else
            if ~quiet
                disp(['Gamma value ',num2str(gam,'%6.1E'),' may be too large.'])
            end
            F = solpath.Fopt(:,:,k-1); % previous solution to current opt. solution
        end
        
        % update the weight matrix W for weighted l1 norm
        if strcmp(options.method, 'wl1')
            eps = 1.e-3;
            Wnew= 1./(abs(F) + eps);
        end
        
        % update the weight matrix for sum of Frobenius norm
        if strcmp(options.method, 'blkwl1')
            % nn is the number of states of each subsystem
            % mm is the number of inputs of each subsystem
            % N is the number of subsystems
            Wnew = ones(sub_mat_size);
            mm = blksize(1);
            nn = blksize(2);
            eps = 1e-3;
            for ii = 1:sub_mat_size(1)
                for jj = 1:sub_mat_size(2)
                    Wnew(ii,jj) = 1 / ( norm( F( mm*(ii-1)+1 : mm*ii, ...
                        nn*(jj-1)+1 : nn*jj ),'fro' ) + eps );
                end
            end
        end
        
        if strcmp(options.method, 'wl1') || strcmp(options.method, 'blkwl1')
            if norm(Wnew - W)/norm(W) < 1.e-2
                if ~quiet
                    disp(['Re-weighted scheme converges in ', num2str(reweightedstep),' steps.'])
                end
                break;
            else
                W = Wnew;
            end
        end
        
    end
    fprintf('Number of iterations in ISTA: %d\n',itr);
    
    
    % record the feedback gain F, H2 norm, and the number of nonzero
    % entries in F
    solpath.gam(k)   = gam;
    solpath.F(:,:,k) = F;
    solpath.J(k)     = trace(B1' * lyap((A - B2*F)', Q + F'*R*F) * B1);
    
    if strcmp(options.method, 'blkwl1') || ...
            strcmp(options.method, 'blkslog') || ...
            strcmp(options.method, 'blkcard')
        solpath.nnz(k) = nnz(F) / ( blksize(1) * blksize(2) );
    else
        solpath.nnz(k) = nnz(F);
    end
    
    % polishing step
    % structural identity IS
    %     IS = double( F ~= 0 );
    %     [solpath.Fopt(:,:,k), solpath.Jopt(k)] = SH2_newtonCG(A,B1,B2,Q,R,IS,F,tolNT);
    
    %fprintf('%4.4f\t\t%4d\t\t%5.2f\n', solpath.gam(k),solpath.nnz(k), solpath.Jopt(k));
    solpath.tBuf(k)=toc(tITAs);
    solpath.itrBuf(k)=itr;
    
end

% save('ista_temp.mat')

end