%% Sparsity-Promoting Linear Quadratic Regulator
%
% Written by Fu Lin, January 2012
%
% Description:
%
% The sparsity-promoting linear quadratic regulator problem is solved using
% the alternating direction method of multipliers (ADMM)
%
%       minimize J(F) + gamma * g(F)             (SP)
%
% where J is the closed-loop H2 norm, g is a sparsity-promoting penalty
% function, and gamma is the parameter that emphasizes the importance of
% sparsity.
%
% After the sparsity pattern has been identified, the structured H2 problem
% is solved
%
%       minimize    J(F)
%                                                                (SH2)
%       subject to  F belongs to identified sparsity pattern
%
% Syntax:
%
% solpath = lqrsp(A,B1,B2,Q,R,options)
%
% Inputs:
%
% (I)  State-space representation matrices {A,B1,B2,Q,R};
%
% (II) Options
%
%     a) options.method specifies the sparsity-promoting penalty function g
%
%        options.method = 'card'     --> cardinality function,
%                       = 'l1'       --> l1 norm,
%                       = 'wl1'      --> weighted l1 norm,
%                       = 'slog'     --> sum-of-logs function,
%                       = 'blkcard'  --> block cardinality function,
%                       = 'blkl1'    --> sum of Frobenius norms,
%                       = 'blkwl1'   --> weighted sum of Frobenius norms,
%                       = 'blkslog'  --> block sum-of-logs function;
%
%     b) options.gamval         --> range of gamma values;
%     c) options.rho            --> augmented Lagrangian parameter;
%     d) options.maxiter        --> maximum number of ADMM iterations;
%     e) options.blksize        --> size of block sub-matrices of F;
%     f) options.reweightedIter --> number of iterations for the reweighted
%                                   update scheme.
%
% The default values of these fields are
%
% options.method         = 'wl1';
% options.gamval         = logspace(-4,1,50);
% options.rho            = 100;
% options.maxiter        = 1000;
% options.blksize        = [1 1];
% options.reweightedIter = 3.
%
% Output:
%
% solpath -- the solution path parameterized by gamma -- is a structure
% that contains:
%
% (1) solpath.F    --> feedback gains resulting from the control problem
%                      (SP);
% (2) solpath.Fopt --> feedback gains resulting from the structured control
%                      problem (SH2);
% (3) solpath.J    --> quadratic performance of the solutions to (SP);
% (4) solpath.Jopt --> optimal quadratic performance of the solutions to
%                      (SH2);
% (5) solpath.nnz  --> number of nonzero elements (blocks) of optimal
%                      sparse (block sparse) feedback gains;
% (6) solpath.gam  --> values of sparsity-promoting parameter gamma.
%
% Reference:
%
% F. Lin, M. Fardad, and M. R. Jovanovic, IEEE Trans. Automat. Control,
% submitted (2011); also arXiv:1111.6188v1
%
% Additional information:
%
% http://www.umn.edu/~mihailo/software/lqrsp/

%% ADMM main function

function solpath = lqrsp(A,B1,B2,Q,R,options)

% Initialization

if nargin < 5
    error('The number of input arguments to lqrsp should be at least five.')
elseif nargin == 5
    options = struct('method','wl1','reweightedIter',3, ...
        'gamval',logspace(-4,1,50),'rho',100,'maxiter',1000,'blksize',[1 1]);
elseif nargin > 6
    error('Too many inputs to lqrsp.')
end

% Data preprocessing
gamval  = options.gamval;
rho     = options.rho;
blksize = options.blksize;

% set the number of reweighted scheme for
% the weighted l1 or the weighted sum of Frobenius norms
if strcmp(options.method, 'wl1') || strcmp(options.method, 'blkwl1')
    reweighted_Max_Iter = options.reweightedIter;
else
    reweighted_Max_Iter = 1;
end

% size of the input matrix
[n,m] = size(B2);

% preallocate memory for output variables
solpath.F    = zeros(m,n,length(gamval));
solpath.nnz  = zeros(1,length(gamval));
solpath.J    = zeros(1,length(gamval));
solpath.gam  = zeros(1,length(gamval));
solpath.Fopt = zeros(m,n,length(gamval));
solpath.Jopt = zeros(1,length(gamval));
solpath.tBuf = zeros(1,length(gamval));
solpath.itrBuf = zeros(1,length(gamval));


% use the centralized gain as the initial condition for ADMM
F = lqr(A,B2,Q,R);
% G = F;
% Lambda = zeros(m,n);

% weight matrix for weighted l1
if strcmp(options.method, 'wl1')
    W = ones(size(F));
end

% weight matrix for sum of Frobenius norm
% sub_mat_size is numbers of rows and columns of partitioned block submatrices
if strcmp(options.method, 'blkwl1')
    sub_mat_size = size(F)./blksize;
    W = ones(sub_mat_size);
end

% absolute and relative tolerances for the stopping criterion of ADMM
eps_abs = 1.e-4;
eps_rel = 1.e-2;

% stopping criterion tolerance for the Anderson-Moore method and Newton's
% method
tolAM = 1.e-2;
tolNT = 1.e-3;

ADMM_Max_Iter = options.maxiter;

%fprintf('%s\t\t%4s\t%10s\n', 'gamma', 'nnz', 'H2_norm');

%% control of display messages
quiet = 1;

%% Solve the sparsity-promoting optimal control problem for each value of
% gamma
for k = 1:length(gamval)
    tADMMs=tic;
    F = lqr(A,B2,Q,R);
    G = F;
    Lambda = zeros(m,n);

    gam = gamval(k);
    
    for reweightedstep = 1 : reweighted_Max_Iter
        
        % Solve the minimization problem using ADMM
        for ADMMstep = 1 : ADMM_Max_Iter
            
            % ========================================================
            % F-minimization step using Anderson-Moore method
            % ========================================================
            U = G - Lambda/rho;
            F = Fmin(A,B1,B2,Q,R,U,rho,F,tolAM);
            %             F = Fmin_L(A,B1,B2,Q,R,U,rho,F,tolAM);
            % ========================================================
            
            % ========================================================
            % G-minimization step
            % ========================================================
            V = F + Lambda/rho;
            
            switch options.method
                case 'l1'
                    % shrinkage operator
                    Gnew = shrinkage(V,ones(size(F)),gam,rho);
                case 'wl1'
                    % shrinkage operator
                    Gnew = shrinkage(V,W,gam,rho);
                case 'slog'
                    % proximity operator for log-sum
                    Gnew = slog(V,gam,rho);
                case 'card'
                    % truncation operator
                    Gnew = trun(V,gam,rho);
                case 'blkl1'
                    % block shrinkage operator
                    sub_mat_size = size(F)./blksize;
                    Gnew = block_shrinkage(V,...
                        ones(size(F)./blksize),gam,rho,blksize,sub_mat_size);
                case 'blkwl1'
                    % block shrinkage operator
                    sub_mat_size = size(F)./blksize;
                    Gnew = block_shrinkage(V,W,gam,rho,blksize,sub_mat_size);
                case 'blkslog'
                    % proximity operator for sum-of-logs
                    sub_mat_size = size(F)./blksize;
                    Gnew = block_slog(V,gam,rho,blksize,sub_mat_size);
                case 'blkcard'
                    % truncation operator
                    sub_mat_size = size(F)./blksize;
                    Gnew = block_trun(V,gam,rho,blksize,sub_mat_size);
            end
            
            if ~isreal(Gnew)
                error('The solution to G-minimization step is not real!')
            end
            
            % dual residual
            resG = norm(G - Gnew,'fro');
            G = Gnew;
            
            % primal residual
            resFG = norm(F - G,'fro');
            % ===========================================================
            
            % ==================== update the dual variable ===============
            Lambda = Lambda + rho * ( F - G );
            % =============================================================
            
            % stoppin criterion for ADMM
            % evaluate the primal epsilon and the dual epsilon
            eps_pri  = sqrt(n*m) * eps_abs + eps_rel * max(norm(F,'fro'), norm(G,'fro'));
            eps_dual = sqrt(n*m) * eps_abs + eps_rel * norm(Lambda,'fro');
            
            if  (resFG < eps_pri)  &&  (rho*resG < eps_dual)
                break;
            end
            
            if ~quiet
                disp([num2str(ADMMstep),'   ',num2str(gam,'%6.1E'),'   ',num2str(resFG,'%6.1E'),'    ',num2str(resG,'%6.1E')])
            end
            
        end
        fprintf('Number of iterations in ADMM: %d\n',ADMMstep);
        solpath.nItr(k) = ADMMstep;
        
        if (ADMMstep == ADMM_Max_Iter) && (~quiet)
            disp('Maximum number of ADMM steps reached!')
            disp(['The primal residual is ', num2str(resFG,'%10.4E')]);
            disp(['The dual residual is ', num2str(rho*resG,'%10.4E')]);
        end
        
        if max( real( eig( A - B2*G ) ) ) < 0
            F = G;
        else
            if ~quiet
                disp(['Gamma value ',num2str(gam,'%6.1E'),' may be too large.'])
            end
            F = solpath.Fopt(:,:,k-1);
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
    IS = double( F ~= 0 );
    [solpath.Fopt(:,:,k), solpath.Jopt(k)] = SH2_newtonCG(A,B1,B2,Q,R,IS,F,tolNT);
    
    %fprintf('%4.4f\t\t%4d\t\t%5.2f\n', solpath.gam(k),solpath.nnz(k), solpath.Jopt(k));
    solpath.tBuf(k)=toc(tADMMs);
    solpath.itrBuf(k)=ADMMstep;    
end

end