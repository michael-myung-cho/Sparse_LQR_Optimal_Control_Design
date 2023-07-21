%% Sparse Anderson-Moore Algorithm
% By Myung (Michael) Cho
%
% ----



function solpath = ISPA_SparseK(A,B1,B2,Q,R,options)

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

if strcmp(options.method, 'blkl1')
    sub_mat_size = [m,n]./blksize;
    mm = blksize(1);
    nn = blksize(2);
end

%% absolute tolerances for the stopping criterion
eps_abs = 1.e-4;
Max_Iter = options.maxiter;


%% display result
quiet = 0; % display option


%% LQR point
F = lqr(A,B2,Q,R);

%% define funtion

max_eig = @(F) max(real(eig(A-B2*F)));
lyap_P = @(F) lyap((A - B2 * F)', Q + F' * R * F);
lyap_L = @(F) lyap((A - B2 * F), B1*B1');
Jobj = @(F) trace(B1' * lyap_P(F) * B1);
gradJ_fun = @(F,P,L) 2*(R*F - B2'*P)*L;

switch options.method
    case 'l0'
        proj_fun = @(F,s) proj_l0(F,s);
    case 'l1'
        proj_fun = @(F,s) proj_l1(F,s);
    case 'blkl1'
        proj_fun = @(F,s) proj_blkl1(F,s,blksize);
end

%% Simulation starts
for k = 1:length(gamval)
    tITAs=tic;
    gam=1;
    s_gamma = gamval(k);
    rho = rhoinit;
    
    for reweightedstep = 1 : reweighted_Max_Iter
        
        %% Solve the minimization problem using Iterative Soft-Thresholding
        % Find a feasible point
        F = proj_fun(F,s_gamma);
        if (max_eig(F) >= 0)
            F = ISTA_feasible_init(A,B1,B2,Q,R,s_gamma,gam,options); % feasible inital point
        end
        
        % iteration starts
        for itr=1:Max_Iter
            stable_flag=1;
            F_pre = F;
            
            P = lyap_P(F_pre);
            L = lyap_L(F_pre);
            
            gradJ = gradJ_fun(F_pre, P, L);
            
            while stable_flag == 1
                F = F_pre - rho*gradJ;  % Gradient descent
                F = proj_fun(F,s_gamma); % projection onto L1 ball
                
                %% stability check
                beta = 0.7;
                if (max_eig(F) >= 0 || Jobj(F) > Jobj(F_pre)) %+ rho * alpha * trace( -gradJ' * gradJ))
                    % backtacking with small step size
                    rho = rho*beta;
                    if rho < 10^-16
                        disp('small step size - break');
                        break;
                    end
                else
                    rho=rhoinit;
                    stable_flag=0;
                end
            end
            
            % stoppin criterion
            resF = norm(F-F_pre,'fro');
            if  (resF < eps_abs && itr > 5 && stable_flag == 0) % converge stop
                %disp('converge stop');
                break;
            end          
        end
        
        %% Display info.
        if (itr == Max_Iter) && (~quiet)
            disp('Maximum number of SPA steps reached!')
            disp(['The residual is ', num2str(resF,'%10.4E')]);
        end
        
        if max_eig(F) >= 0
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
    fprintf('Number of iterations in ISPA: %d\n',itr);
    
    % record the feedback gain F, H2 norm, and the number of nonzero
    % entries in F
    solpath.gam(k)   = gam;
    solpath.F(:,:,k) = F;
    solpath.J(k)     = trace(B1' * lyap_P(F) * B1);
    
    if strcmp(options.method, 'blkwl1')
        solpath.nnz(k) = nnz(F) / ( blksize(1) * blksize(2) );
    else
        solpath.nnz(k) = nnz(F);
    end
    
    solpath.tBuf(k)=toc(tITAs);
    solpath.itrBuf(k)=itr;
    
end
%save('ista_temp.mat')

end


%% Projection onto L1 ball
function F = proj_l1(F,s_gamma)
[m,n]=size(F);
if norm(F(:),1) > s_gamma
    sort_F = sort(abs(F(:)),'descend');
    sum_F=0;
    for ii=1:n*m
        sum_F = sum_F + sort_F(ii);
        if sort_F(ii) - (sum_F - s_gamma)/ii > 0
            max_ind=ii;
        end
    end
    
    thre= (sum(sort_F(1:max_ind))-s_gamma)/max_ind;
    F = sign(F).*((abs(F)>thre).*(abs(F)-thre));
end
end

%% Projection onto L0 ball
function F = proj_l0(F_temp,nnz_val)
    [m,n]=size(F_temp);
    [~,ind] = sort(abs(F_temp(:)),'descend');
    F=zeros(m,n);
    F(ind(1:nnz_val))=F_temp(ind(1:nnz_val));
end



%% Projection onto block L1 ball
function F = proj_blkl1(F,s_gamma,blksize)
[m,n]=size(F);
sub_mat_size = [m,n]./blksize;
U = zeros(sub_mat_size);
mm = blksize(1);
nn = blksize(2);
G_value=0;
for i = 1:sub_mat_size(1)
    for j = 1:sub_mat_size(2)
        Uij=F(mm*(i-1)+1:mm*i, nn*(j-1)+1:nn*j);
        U(i,j)=norm( Uij, 'fro' );
        G_value=G_value+U(i,j);
    end
end
if G_value > s_gamma
    sort_U = sort(U(:),'descend');
    sum_U=0;
    for ii=1:sub_mat_size(1)*sub_mat_size(2)
        sum_U=sum_U+sort_U(ii);
        if sort_U(ii) - (sum_U - s_gamma)/ii > 0
            max_ind=ii;
        end
    end
    thre=(sum(sort_U(1:max_ind))-s_gamma)/max_ind;
    for i = 1:sub_mat_size(1)
        for j = 1:sub_mat_size(2)
            if U(i,j) <= thre
                F(mm*(i-1)+1:mm*i, nn*(j-1)+1:nn*j)=0;
            else
                F(mm*(i-1)+1:mm*i, nn*(j-1)+1:nn*j)=(U(i,j)- thre)*F(mm*(i-1)+1:mm*i, nn*(j-1)+1:nn*j)/U(i,j);
            end
        end
    end
end
end
