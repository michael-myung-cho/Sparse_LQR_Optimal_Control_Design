% Spatially distributed systems
% An example from Motee and Jadbabaie, 
% "Optimal control of spatially distributed systems", 
% IEEE Trans. Automat. Control, 
% vol. 53, no. 7, pp. 1616–1629, 2008.


clear; clc;
%profile on

addpath('ADMM');
addpath('ISTA');
addpath('ISPA');


% N nodes randomly distributed in a box [0 L] x [0 L]
N = 50;
L = 100;

%load ("./examples/positions.mat")
% or generate the random distribution of the nodes
pos = L*rand(N,2);

% the state-space representation of each system i
Aii = [1 1; 1 2];
Bii = [0; 1];
n   = size(Bii,1);
Aij = eye(n);

B1  = kron(eye(N), Bii);
B2  = B1;

% construct 
% (a) the Euclidean distance matrix that describes the distance
% between two systems i and j
% (b) the A-matrix of the distributed system

A = zeros(2*N,2*N);
dismat = zeros(N,N);
for i = 1:N
    for j = i:N        
        if i == j
            A( (i-1)*n + 1 : i*n, (j-1)*n + 1 : j*n ) = Aii;
        else            
            dismat(i,j) = sqrt( norm( pos(i,:) - pos(j,:) )^2 );
            dismat(j,i) = dismat(i,j);               
            A( (i-1)*n + 1 : i*n, (j-1)*n + 1 : j*n ) = Aij / exp( dismat(i,j) );
            A( (j-1)*n + 1 : j*n, (i-1)*n + 1 : i*n ) = Aij / exp( dismat(j,i) );
        end
    end
end

% state and control penalty weight matrices
Q = eye(2*N);
R = eye(N);

%% compute sparse feedback gains
fprintf('ADMM \n');
options_admm = struct('method','l1','gamval',[100:100:1000],'rho',100,'maxiter',100000,'blksize',[1 1],'reweightedIter',1);
tADMMs=tic;
solpath_admm = lqrsp(A,B1,B2,Q,R,options_admm);
tADMMe=toc(tADMMs);


%%%%%%%% ISTA %%%%%%
fprintf('ISTA \n');
options_ista = struct('method','l1','gamval',[100:100:1000],'rho',100,'maxiter',100000,'blksize',[1 1],'reweightedIter',1);
tITAs=tic;
solpath_ista = ISTA_SparseK(A,B1,B2,Q,R,options_ista);
tITAe=toc(tITAs);


%%%%%% Sparsity-Projection Algorithm  %%%%%%
% fprintf('SPA \n');
% options_SPA = struct('method','l1','gamval',[10,30,50],'rho',20,'maxiter',10000,'blksize',[1 3],'reweightedIter',1);
% tSAMs=tic;
% solpath_SPA = SPA_ver01(A,B1,B2,Q,R,Finit,solpath_ista.F,options_SPA);
% tITAe=toc(tSAMs);



% Computational Results
%% number of nonzero blocks vs. gamma
figure(10)
semilogx(solpath_admm.gam,solpath_admm.nnz,'b-o','MarkerSize',10,'LineWidth',2)
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
% semilogx(solpath_SPA.gam,solpath_SPA.nnz,'g-s','MarkerSize',15,'LineWidth',3)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 45)
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 45)



%% complexity
figure(2)
%bar([solpath_admm.tBuf;solpath_ista.tBuf;solpath_SPA.tBuf]')
bar([solpath_admm.tBuf;solpath_ista.tBuf]')

%% J value
figure(3)
%bar([solpath_admm.J;solpath_ista.J;solpath_SPA.J]')
bar([solpath_admm.J;solpath_ista.J]')



% number of nonzeros vs. gamma
% figure
% plot(solpath.gam, solpath.nnz, 'o', 'LineWidth', 2, 'MarkerSize', 10)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18, 'xscale', 'log')
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 26)
% 
% % H2 performance vs. gamma
% [Fc, P] = lqr(A,B2,Q,R);
% Jc = trace(P*(B1*B1'));
% 
% figure
% semilogx(solpath.gam,(solpath.Jopt - Jc)/Jc*100,...
%     'r+','LineWidth',2,'MarkerSize',10)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18, 'xscale', 'log')
% xlab = xlabel('\gamma','interpreter', 'tex');
% set(xlab, 'FontName', 'cmmi10', 'FontSize', 26)
% set(gca,'YTick',0:10:60,'YTickLabel',{'0%','10%','20%','30%','40%','50%','60%'})
% 
% % Sparsity vs. H2 performance
% figure
% plot(solpath.nnz/nnz(Fc)*100,(solpath.Jopt - Jc)/Jc*100,...
%     'r+','LineWidth',2,'MarkerSize',10)
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18)
% set(gca,'YTick',0:10:60,'YTickLabel',{'0%','10%','20%','30%','40%','50%','60%'})
% set(gca,'XTick',0:20:100,'XTickLabel',{'0%','20%','40%','60%','80%','100%'})
% 
% % Communication architecture of the distributed controller
% 
% F_idx = [39,43,48];
% 
% for kk = 1:length(F_idx)
% 
%     % assign a sparse feedback gain matrix
%       F = solpath.F(:,:,F_idx(kk));
%     idx = zeros(2,nnz(F));
% 
%     % count the number of nonzero 1x2 blocks in F
%     % and record the indices of the nonzero blocks
%     k = 0;
%     for i = 1:N
%         rowF = F(i,:);
%         for j = 1:N
%             if norm( rowF( 2*(j-1) + 1 : 2*j ) ) ~= 0 && i ~= j
%                 k = k + 1;
%                 idx( :, k ) = [i j]';
%             end
%         end
%     end
% 
%     % number of links between systems
%     m = nnz(idx)/2;
%     
%     % remove those extra zeros in idx
%     idx = idx(:,1:m);
% 
%     % drawing the communication graphs
% 
%     % figure number
%     nn = 100 + kk;
%     figure(nn)
%     hold on
% 
%     % draw the communication links
%     ii = sqrt(-1);
%     for k = 1:m
%         i = idx(1,k);
%         j = idx(2,k);
%         figure(nn),
%         plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'r', 'LineWidth', 1 );
%     end
% 
%     % draw the nodes (systems)
%     figure(nn),
%     plot( pos(:,1) + pos(:,2)*sqrt(-1),'o','LineWidth',2,'MarkerSize',10)
%     hold off;
%     h = get(gcf,'CurrentAxes');
%     set(h, 'FontName', 'cmr10', 'FontSize', 18)    
%     
% end
% 
% solpath.gam(F_idx)
% solpath.nnz(F_idx)/nnz(Fc)
% (solpath.Jopt(F_idx)-Jc)/Jc
% 
% % stability of truncated centralized feedback gains
% 
% for k = 1:length(solpath.gam)
%     
%     % the number of nonzero elements of sparse feedback gains
%     nzF  = solpath.nnz(k);
%     % sort all elements of Fc according to their absolute values
%     stFc = sort(vec(abs(Fc)),'descend');
%     % find the threshold value for the truncation
%     threshold = stFc(nzF);
%     % truncate the centralized gain
%     trunFc = Fc .* double(abs(Fc) >= threshold);
%     % if the truncated gain is non-stabilizing,
%     % assign a negative value to J; otherwise, compute the H2 norm
%     if max( real( eig( A - B2 * trunFc ) ) ) > 0
%         break;    
%     end
% end
% 
% nnz(trunFc) / nnz(Fc)
% 
% % plot the sparsity pattern of the non-stabilizing truncated gain
% figure,spy(trunFc,10)
% xlabel('')
% h = get(gcf,'CurrentAxes');
% set(h, 'FontName', 'cmr10', 'FontSize', 18) 
% 
% % ===== plot the communication architecture of truncated controller =======
% F = trunFc;
% idx = zeros(2,nnz(F));
%     % count the number of nonzero 1x2 blocks in F
%     % and record the indices of the nonzero blocks
%     k = 0;
%     for i = 1:N
%         rowF = F(i,:);
%         for j = 1:N
%             if norm( rowF( 2*(j-1) + 1 : 2*j ) ) ~= 0 && i ~= j
%                 k = k + 1;
%                 idx( :, k ) = [i j]';
%             end
%         end
%     end
% 
%     % number of links between systems
%     m = nnz(idx)/2;
%     
%     % remove those extra zeros in idx
%     idx = idx(:,1:m);
% 
%     % drawing the communication graphs
% 
%     % figure number    
%     nn = 1000;
%     figure(nn)
%     hold on
% 
%     % draw the communication links
%     ii = sqrt(-1);
%     for k = 1:m
%         i = idx(1,k);
%         j = idx(2,k);
%         figure(nn),
%         plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'r', 'LineWidth', 1 );
%     end
% 
%     % draw the nodes (systems)
%     figure(nn),
%     plot( pos(:,1) + pos(:,2)*sqrt(-1),'o','LineWidth',2,'MarkerSize',10)
%     hold off;
%     h = get(gcf,'CurrentAxes');
%     set(h, 'FontName', 'cmr10', 'FontSize', 18)    
