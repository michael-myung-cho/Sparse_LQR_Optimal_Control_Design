function result= GraSPutil_ComputeCentra(sys_para,Q,options, sp_cons, IS)
% IS: position of 1 is the position of local links
% options.tol
% options.tolPolish
% options.outMax
% options.method

% time:
%
% 1. newton minimization (including gradient sort and pruning)
% 2. backtracking if unstable
% 3. polish

debugmod = 0;


R = sys_para.R;
A = sys_para.A;
B=sys_para.B2;
B_tilde=sys_para.B1;


[n,m] = size(B);
epsabs = 1e-4;
eps_rel = 1e-4;
struct_inter = ones(m,n)-IS; % structure of inter-area links

timeInfo.MinStep = zeros(1,length(sp_cons)); % time spent minimization step for each sp_cons(k), including pruning
timeInfo.backTrack = zeros(1,length(sp_cons)); % time spent in backtracking when unstable for each sp_cons(k)
timeInfo.polish = zeros(1,length(sp_cons)); % time spent in polishing step for each sp_cons(k)
timeInfo.nStep = zeros(1,length(sp_cons)); % number of iterations for each sp_cons(k)
timeInfo.nCG_polish = zeros(1,length(sp_cons)); % number of conjugate gradient iterations in polishing step for each sp_cons(k)
timeInfo.nPolish = zeros(1,length(sp_cons)); % number of newton steps to polish
timeInfo.nCG = zeros(1,length(sp_cons)); % number conjuate gradient steps (tot)


Kx= lqr(sys_para.A,sys_para.B2,Q,sys_para.R);

% GraSP
for k = 1:length(sp_cons)
    tic;
    tGraSPs=tic; 
    
    s = sp_cons(k);
    if options.initChoice == 1 % same initial value for all s
        Kx = K0;
    end
    
    K_prev = Kx; 
    
    if max(real(eig(A-B*Kx))) > 0
        error('Directly truncating not stable\n');
    end    
    
    for itr = 1:options.outMax
        %% minimization step

        % initial value: last Kx
        % gradient
        Ltemp = lyap(A-B*Kx,B_tilde*B_tilde.');
        Ptemp = lyap((A-B*Kx).',Q+Kx.'*R*Kx);
        J_prev = trace(B_tilde'*Ptemp*B_tilde); % objective before descent
        grad = (2*(R*Kx-B.'*Ptemp)*Ltemp);
        
        % 2s support: maximum gradient
        
        [sorted_values, sorted_index] = sort(abs(vec(grad.*(struct_inter))),'descend');
        Z_index = sorted_index(1:min(2*s,nnz(sorted_values)));
        Z_index_matrix = zeros(size(grad));
        Z_index_matrix(Z_index) = 1;
        % merge support
        T_index_matrix = Z_index_matrix | IS | Kx~=0;
        % take one newton step over support
        
        if debugmod
            fprintf('computing newton direction...\n');
        end
        if strcmp(options.method,'newtonCD')
            Fnt = newtonCD(A,B,Kx,Ptemp,Ltemp,R,T_index_matrix, grad.*T_index_matrix);
        else
            [Fnt, CGtemp] = newtonCG(A,B,Kx,Ptemp,Ltemp,R,grad.*T_index_matrix,T_index_matrix);
            timeInfo.nCG(k) = timeInfo.nCG(k) + CGtemp +1;
        end
        [J_temp, Fnext, ~] = ArmijoUpdate(A, B_tilde, B, Q,R, Kx, Fnt, grad.*T_index_matrix);
        
        % check stability before pruning
        if max(real(eig(A-B*Fnext)))>=0
            error('not stable before prune');
        end
        
        % prune: only keep s inter area links, but keep stable
        if debugmod
            fprintf('pruning...\n');
        end
        
        [~, sorted_index] = sort(abs(vec(Fnext.*(struct_inter))),'descend');
        s_index = sorted_index(1:min(s,nnz(struct_inter)));
        s_index_matrix = zeros(size(Fnext));
        s_index_matrix(s_index) = 1;
        
        % s_index union diagonal elements
        Fnext((s_index_matrix | IS) ==0) = 0;
        timeInfo.MinStep(k) = timeInfo.MinStep(k) + toc;
        
        %% backtracking if unstable
        count = 1;
        s_hat = s;
        while max(real(eig(A-B*Fnext)))>=0
            fprintf('unstable after pruning, count %d\n',count);
            if s_hat > nnz(Kx.*~IS)
                s_hat = s_hat-1;
                [sorted_values, sorted_index] = sort(abs(vec(grad.*(struct_inter))),'descend');
                Z_index = sorted_index(1:min(2*s_hat,nnz(sorted_values)));
                Z_index_matrix = zeros(size(grad));
                Z_index_matrix(Z_index) = 1;
                % merge support
                T_index_matrix = Z_index_matrix | IS | Kx~=0;
                [Fnt, CGtemp] = newtonCG(A,B,Kx,Ptemp,Ltemp,R,grad.*T_index_matrix,T_index_matrix);
                timeInfo.nCG(k) = timeInfo.nCG(k) + CGtemp +1;
                [J_temp, Fnext, ~] = ArmijoUpdate(A, B_tilde, B, Q,R, Kx, Fnt, grad.*T_index_matrix);
                % prune
                [~, sorted_index] = sort(abs(vec(Fnext.*(struct_inter))),'descend');
                s_index = sorted_index(1:min(s_hat,nnz(struct_inter)));
                s_index_matrix = zeros(size(Fnext));
                s_index_matrix(s_index) = 1;
                Fnext((s_index_matrix | IS) ==0) = 0;
            else
                % return to last structure of this sp_cons(k) and descent
                % in supp(Kx) direction
                direction = double(Kx ~=0);
                [Fnt, CGtemp] = newtonCG(A,B,Kx,Ptemp,Ltemp,R,grad.*direction,direction);
                timeInfo.nCG(k) = timeInfo.nCG(k) + CGtemp +1;
                [J_temp, Fnext] = ArmijoUpdate(A, B_tilde, B, Q,R, Kx, Fnt, grad.*direction);
%                 if j == 1
%                     fprintf('constraint not active \t');
%                 end
                break;
            end
            
            count = count + 1;
        end
        
        
        % compute Jopt
        
        Ptemp = lyap((A - B * Fnext).',Q+Fnext' * R * Fnext);
        Jopt = trace(B_tilde'*Ptemp*B_tilde);
        assert(nnz(Fnext.*~IS)<=s);
        Kx =  Fnext;
%         if Jopt <= J_prev % if decent causes energy to decrease
%             Kx = Fnext;
%         else
%             Jopt = J_prev;
%         end
        
        if max(real(eig(A-B*Kx))) > 0
            error('unstable');
        end
        
        timeInfo.backTrack(k) = timeInfo.backTrack(k) + toc;
        
        if options.ADMMcomp == 1
            resNorm = norm(Kx - K_prev,'fro'); % absolute residue
            tolADMM = sqrt(n*m) * epsabs + eps_rel * norm(Kx,'fro');
            tol = tolADMM;
            if debugmod
                [k, itr, resNorm, tol]
            end
        else
            resNorm = norm(Kx - K_prev,'fro');%/norm(Kx,'fro');
            tol = options.tol;
            if debugmod
                [k, itr, resNorm]
            end
        end
        
        if itr>1 && resNorm < tol
            % polishing according to the indentified structur
            if debugmod
                fprintf('polishing...\t');
            end
            if strcmp(options.method,'newtonCD')
                [Kx, Jopt,timeInfo.nPolish(k), timeInfo.nCG_polish(k)]= SH2_newtonCD_slow(A,B_tilde,B,Q,R,double(Kx~=0),Kx.*(double(Kx~=0)),options.tolPolish);
            else
                [Kx, Jopt,timeInfo.nPolish(k), timeInfo.nCG_polish(k)]= SH2_newtonCG(A,B_tilde,B,Q,R,double(Kx~=0),Kx.*(double(Kx~=0)),options.tolPolish);
                timeInfo.nCG(k) = timeInfo.nCG(k) + timeInfo.nCG_polish(k) +1;
            end
            timeInfo.polish(k) = toc;
            break;
        end
        K_prev = Kx;
        
    end
    fprintf('Number of iterations in GraSP: %d\n',itr);
    result.tBuf(k)=toc(tGraSPs);
    result.itrBuf(k)=itr;
    
    timeInfo.nStep(k) = itr;
    result.Jopt(k) = Jopt;
    result.Ksparse(:,:,k) = Kx;
    result.cardn(k) = nnz(Kx.*struct_inter);
    
end
end

function F = proj_l0(F_temp,nnz_val)
[m,n]=size(F_temp);
[~,ind] = sort(abs(F_temp(:)),'descend');
F=zeros(m,n);
F(ind(1:nnz_val))=F_temp(ind(1:nnz_val));
end