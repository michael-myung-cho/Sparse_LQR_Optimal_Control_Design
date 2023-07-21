%% Multi-agent control scenario
% K_sparse matrix dataset generation
% Dataset {x,finit,y}

clear;
clc;

addpath('../../ISTA');

%% State-space representation (Block sparsity example)
% number of systems
N = 10;

% size of each system
nn = 3; 
mm = 1;

% use cyclic condition to obtain unstable system
a = ones(1,nn);
b = 1.5*(sec(pi/nn))*a;

% state-space representation of each system
Aa = -diag(a) + diag(b(2:nn),-1); 
Aa(1,nn) = -b(1);
Bb1 = diag(b);
Bb2 = zeros(nn,2); % MCho
Bb2(1) = b(1);      
Bb2(2,2) = b(1); % MCho   

% non-symmetric weighted Laplacian matrix 

% adjacency matrix 
Ad = toeplitz([1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]);
for i = 1 : N
    for j = 1 : N
        if i ~= j
            cij = 0.5 * ( i - j );
        else
            cij = 0;
        end
        Ad( nn*(i-1)+1 : nn*i, nn*(j-1)+1 : nn*j) = cij * eye(nn);
    end
end
        
% take the sum of each row
d = sum(Ad,2);

% form the Laplacian matrix
L = Ad - diag(d);

% state-space representation of the interconnected system

A  = kron(eye(N), Aa) - L;
B1 = kron(eye(N), Bb1);
B2 = kron(eye(N), Bb2);
Q  = eye(nn*N);
R  = eye((nn-1)*N);


% normalize matrix
% A_norm = A/norm(A,'fro');
% B1_norm = B1/norm(A,'fro');
% B2_norm = B2/norm(A,'fro');

F_LQR = lqr(A,B2,Q,R);
% F_LQR_norm = lqr(A_norm,B2_norm,Q,R);

% Compute the optimal sparse feedback gains
options = struct('method','l1','gamval', 1,'rho',100,'maxiter',10000,'blksize',[1 3],'reweightedIter',1);


%% training (1000 ea) & test (100 ea) data gen
nTrainSamp=1000;
nTestSamp=nTrainSamp*0.1;
nSamp=nTrainSamp+nTestSamp;
x_data=[]; % x_data = [A,B1,B2]

y_data=[]; % K sparese
% y_data_normalized=[];

Finit_data=[];
% Finit_data_normalized=[];
normalized_factor=sqrt(norm(A,'fro')^2+norm(B2,'fro')^2);
A_nor = A;%/normalized_factor;
B1_nor = B1;
B2_nor = B2;%/normalized_factor;

for iSamp=1:nSamp
    iSamp
    
    Noise=(abs(A_nor)>10^-6).*(randn(N*nn,N*nn)*0.1);
 
    F_init = lqr(A_nor+Noise,B2_nor,Q,R);
    solpath = ISTA_SparseK(A_nor+Noise,B1_nor,B2_nor,Q,R,options);

    A_nor_noise = A_nor+Noise;
    buf=[A_nor_noise(:);B1_nor(:);B2_nor(:)]';
    x_data=[x_data;buf];
    
    Finit_data=[Finit_data;F_init(:)'];
%     Finit_data_normalized=[Finit_data_normalized,;vec(Finit)'/norm(Finit,'fro')];
    
    F_star = solpath.F;
    y_data=[y_data;F_star(:)'];
%     y_data_normalized=[y_data_normalized;vec(solpath.F)'/norm(solpath.F,'fro')];
end

x_train_data=x_data(1:nTrainSamp,:);
y_train_data=y_data(1:nTrainSamp,:);
Finit_train_data=Finit_data(1:nTrainSamp,:);

%x_train_data_normalized=x_data(1:nTrainSamp,:);
%y_train_data_normalized=y_data_normalized(1:nTrainSamp,:);
%Finit_train_data_normalized=Finit_data_normalized(1:nTrainSamp,:);



Finit_test_data=Finit_data(nTrainSamp+1:nTrainSamp+nTestSamp,:);
y_test_data=y_data(nTrainSamp+1:nTrainSamp+nTestSamp,:);
x_test_data=x_data(nTrainSamp+1:nTrainSamp+nTestSamp,:);
%x_test_data_normalized=x_data(nTrainSamp+1:nTrainSamp+nTestSamp,:);
%Finit_test_data_normalized=Finit_data_normalized(nTrainSamp+1:nTrainSamp+nTestSamp,:);
%y_test_data_normalized=y_data_normalized(nTrainSamp+1:nTrainSamp+nTestSamp,:);

filename=strcat('Ksparse_x_f_y_dataset_N10_',num2str(nTrainSamp),'.mat');
save(filename)
