function [Jtemp, Ftemp, stop] = ArmijoUpdate(A, B1, B2, Q,R,F, Fnt, gradF)
% step size search
% energy of current F
% output:
% Jtemp: updated energy
% Ftemp: udpated feedback
% stop ==1 -> cannot descend further
Ptemp = lyap((A - B2 * F).',Q+F' * R * F);
J = trace(B1'*Ptemp*B1);
stop = 0;
stepsize = 1;
if max( real( eig( A - B2 * F ) ) )>0
    error('Initial value not stabilizing');
end
while 1
    
    Ftemp = F + stepsize * Fnt;
    maxEigAcl = max( real( eig( A - B2 * Ftemp ) ) );
    
    if maxEigAcl > 0
        Jtemp = nan;
    else
        Ltemp = lyap( A - B2 * Ftemp, B1 * B1' );
        Jtemp = trace( Ltemp * ( Q + Ftemp' * R * Ftemp ) );
    end
    
    % Armijo rule data
    alpha = 0.3;
    beta  = 0.5;
    if  ~isnan(Jtemp) && J - Jtemp > stepsize * alpha * trace( -Fnt' * gradF)
        if trace( -Fnt' * gradF) < 0
            error('Not a descent direction!\n');
        end
        break;
    end
    
    stepsize = beta*stepsize;
    if ~isnan(Jtemp) && stepsize < 1.e-16
        fprintf('Extremly small stepsize in polishing step!\n');
        Ftemp = F;
        Jtemp = J;
        stop = 1;
        break; %avoid error  
    end
    
    if isnan(Jtemp) && stepsize < 1.e-16
        fprintf('Not stable and Extremly small stepsize in polishing step!\n');
%         stepsize = 0;
        Ftemp = F;
        Jtemp = J;
        stop = 1;
        break; %avoid error  
    end
    
end

if stepsize < 1.e-16 && (trace( -Fnt' * gradF) < 1e-6 || Jtemp >= J)
    fprintf('Not a descent direction\n');
    Ftemp = F;
    Jtemp = J;
    stop = 1;
%     stepsize = 0;
%     error(['The norm of gradient is ', num2str(gradF), '.']);
    % return to previous solution, discard Ftemp
    
end

end