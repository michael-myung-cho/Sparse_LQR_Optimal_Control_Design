function Q= GenQ_consensus(NodesPerArea,StPerGen,Dwgt,Owgt,Rwgt,reg,opt)

% Generate Q matrix for high order model (PST manual)
% M1: M matrix for delta(diagonal)
% M2: M matrix for omega(diagonal)
% WgtRest: weight of rest of the states
% Dwgt: weight of the Delta term
% Owgt: weight of the omega term
% x^T Q x = Dwgt*delta^T* Lunif *delta^T + Owgt*omega^T* Lunif* omega
% + StateRemain^T Rwgt StateRemain + reg*|x|^2

quiet=1;


%for centralized solution only
Ntotal=sum(NodesPerArea); % total number of generators
if strcmp(opt,'all')
    % global consensus
L = Ntotal*eye(Ntotal,Ntotal)-ones(Ntotal,1)*ones(1,Ntotal);

elseif strcmp(opt,'intra')
    L = [];
    for i = 1:length(NodesPerArea)
        nGen = NodesPerArea(i);
        L = blkdiag(L, nGen*eye(nGen,nGen)-ones(nGen,1)*ones(1,nGen) );
    end
    if ~quiet
        fprintf('L:\n');
        disp(L);
        figure
        spy(L)
    end
elseif strcmp(opt,'inter')
    L_all = Ntotal*eye(Ntotal,Ntotal)-ones(Ntotal,1)*ones(1,Ntotal);
    L_intra = [];
    for i = 1:length(NodesPerArea)
        nGen = NodesPerArea(i);
        L_intra = blkdiag(L_intra, nGen*eye(nGen,nGen)-ones(nGen,1)*ones(1,nGen) );
    end
    L = L_all - L_intra;
    if ~ quiet
        disp(L);
    end
else
    error('no such option, either all or intra');
end


Qtemp=blkdiag(Dwgt*L,Owgt*L,Rwgt*eye(sum(StPerGen)-2*Ntotal)); %before permutaion theta1...thetaM | omega1...omegaM
%% take the index of delta, omega, and remaining
IndSt1 = cumsum(StPerGen)-StPerGen+1; %index of delta delta
IndSt2 = IndSt1+1; %index of delta omega
IndRest = 1:sum(StPerGen);
IndRest([IndSt1.' IndSt2.']) = [];
% permutation matrix from original permutation to pmt of Qtemp
pmt_from = zeros(sum(StPerGen),sum(StPerGen));
indices = sub2ind(size(pmt_from),1:sum(StPerGen) ,[IndSt1.' IndSt2.' IndRest]);
pmt_from(indices) = 1;
pmt_to = pmt_from.'; % row permutation matrix

if ~quiet
    fprintf('Before permutation: Q matrix\n')
    disp(Qtemp);
    figure
    spy(Qtemp);
end

%permute column
Qtemp = pmt_to*(Qtemp.');
%permute row
Q=pmt_to*Qtemp.';

% regularize on all terms
Q = Q + reg*eye(size(Q));

if ~quiet
    fprintf('After permutation: Q matrix\n')
    figure
    spy(Q);
    disp(Q);
end

end