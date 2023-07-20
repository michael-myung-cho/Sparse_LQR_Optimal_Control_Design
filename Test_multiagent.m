%% Block sparsity example

clear; clc; clf;
%profile on

addpath('ADMM');
addpath('ISTA');
addpath('ISPA');
addpath('GraSP');

%% Problem setting
% number of systems
N = 5;

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


%% compute block sparse feedback gains

%%%%%% ADMM %%%%%% 
fprintf('ADMM \n');
options_admm = struct('method','l1','gamval', 0.1:0.2:5,'rho',100,'maxiter',10000,'blksize',[1 3], 'reweightedIter',1);
tADMMs=tic;
solpath_admm = lqrsp(A,B1,B2,Q,R,options_admm);
tADMMe=toc(tADMMs);

%%%%%%%% ISTA %%%%%%
fprintf('ISTA \n');
options_ista = struct('method','l1','gamval', 0.1:0.2:5,'rho',100,'maxiter',100000,'blksize',[1 3],'reweightedIter',1);
tITAs=tic;
solpath_ista = ISTA_SparseK(A,B1,B2,Q,R,options_ista);
tITAe=toc(tITAs);


%% dataset save
% Finit = lqr(A,B2,Q,R);
% 
% buf=[vec(A);vec(B1);vec(B2)];
% x_data=buf;
% Finit_data=vec(Finit);
% y_data=vec(solpath_blkwl1_ista.F');
%     
% nTrainSamp=1;
% x_train_data=x_data(:,1:nTrainSamp);
% Finit_train_data=Finit_data(:,1:nTrainSamp);
% y_train_data=y_data(:,1:nTrainSamp);
%     
% filename=strcat('block_sparsity_x_f_y_dataset_',num2str(nTrainSamp),'.mat');
% save(filename)
%% Computational results

%% number of nonzero blocks vs. gamma
figure(1)
semilogx(solpath_admm.gam,solpath_admm.nnz,'b-o','MarkerSize',15,'LineWidth',3)
h = get(gcf,'CurrentAxes');
set(h, 'FontName', 'cmr10', 'FontSize', 18)
xlab = xlabel('\gamma','interpreter', 'tex');
set(xlab, 'FontName', 'cmmi10', 'FontSize', 26)

hold on
semilogx(solpath_ista.gam,solpath_ista.nnz,'r-x','MarkerSize',15,'LineWidth',3)
h = get(gcf,'CurrentAxes');
set(h, 'FontName', 'cmr10', 'FontSize', 45)
xlab = xlabel('\gamma','interpreter', 'tex');
set(xlab, 'FontName', 'cmmi10', 'FontSize', 45)


% hold on
% semilogx(solpath_SDP.gam,solpath_SDP.nnz,'r-x','MarkerSize',15,'LineWidth',3)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 45)
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 45)



%% complexity
figure(2)
bar([solpath_admm.tBuf;solpath_ista.tBuf]')
 
%% J value
figure(3)
bar([solpath_admm.J;solpath_ista.J]')

%% H2 performance vs. gamma
% figure(2)
% semilogx(solpath_blkwl1.gam,solpath_blkwl1.Jopt,'r+','MarkerSize',10,'LineWidth',2)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 26)
% 
% figure(20)
% semilogx(solpath_blkwl1_ista.gam,solpath_blkwl1_ista.Jopt,'r+','MarkerSize',10,'LineWidth',2)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 26)

%% Compare the sparse feedback gains 

%% block sparse feedback gain
% idx_blk = 46;
% solpath_blkwl1.gam(idx_blk)
% solpath_blkwl1.nnz(idx_blk)
% 
% figure(3);
% spy(solpath_blkwl1.Fopt(:,:,idx_blk),30)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% xlabel('')
% 
% 
% solpath_blkwl1_ista.gam(idx_blk)
% solpath_blkwl1_ista.nnz(idx_blk)
% 
% figure(30);
% spy(solpath_blkwl1_ista.Fopt(:,:,idx_blk),30)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% xlabel('')

%% element sparse feedback gain
% idx = 33;
% solpath_wl1.gam(idx)
% solpath_wl1.nnz(idx)
% figure(4)
% ,spy(solpath_wl1.Fopt(:,:,idx),30)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% xlabel('')
% 
% (solpath_blkwl1.Jopt(idx_blk) - solpath_wl1.Jopt(idx))/solpath_wl1.Jopt(idx)

% profile viewer
% profsave