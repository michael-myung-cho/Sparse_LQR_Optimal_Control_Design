

% Use Newton's method in conjunction with conjugate gradient scheme
% to solve the structured H2 problem
%
%      minimize    J(F)
%      subject to  F \in structure
%
% Syntax:
% [Fopt,Jopt] = SH2_newtonCG(A,B1,B2,Q,R,IS,F0,tol);
%
% Inputs: 
% state-space representation data {A,B1,B2,Q,R},
% structural identity matrix IS,
% initial condition F0,
% stopping criterion tolerance tol.
%
% Outputs: 
% minimizer Fopt and minimum value Jopt.
%

function [Fopt,Jopt, NTstep, numCD] = SH2_newtonCD(A,B1,B2,Q,R,IS,F0,tol)
% NTstep: number of newton steps to polish
% numCG: number of conjugate gradient steps

    % check if the initial condition is stabilizing
    if max( real( eig( A - B2*F0 ) ) ) > 0
        error('The initial condition F0 is not a stabilizing gain!')
    end

    F = F0;
    
    % controllability gramain
    L = lyap( (A - B2 * F), B1 * B1' );     
    
    % compute the objective
    J = trace( L * ( Q + F' * R * F ) );
    
    % maximum number of Newton iterations
    NT_Max_Iter = 100;
 
    % control of display messages
    quiet = 0;
    
    % number of conjugate gradient steps used in newton
    numCD = 0;

    for NTstep = 1 : NT_Max_Iter
        
        % compute the gradient 
             P = lyap( (A - B2 * F)', Q + F' * R * F );
         gradF = 2 * ( ( ( R * F - B2' * P ) * L ) .* IS );
        ngradF = norm( gradF, 'fro' );

        if ~quiet
            disp([num2str(NTstep),'   ',num2str(ngradF,'%6.1E')])
        end
        
        if ngradF < tol
            break;
        end

        % compute the Newton direction using the conjugate gradient scheme
%         [Fnt, CDstep ]= newtonCD(A,B2,F,P,L,R,IS);
        [Fnt, CDstep ]= newtonCD_fast(A,B2,F,P,L,R,IS);
        numCD = numCD+CDstep;
        
        stepsize = 1;

        % line search
        stop = 0;
        while 1

                Ftemp = F + stepsize * (Fnt);
            maxEigAcl = max( real( eig( A - B2 * Ftemp ) ) );
            
            if maxEigAcl > 0
                Jtemp = nan;
            else
                Ltemp = lyap( A - B2 * Ftemp, B1 * B1' );                
                Jtemp = trace( Ltemp * ( Q + Ftemp' * R * Ftemp ) );
            end
            
            if Jtemp > J
                error('not a descend direction')
            end

            % Armijo rule data
            alpha = 0.3;
            beta  = 0.5;
            
            if  ~isnan(Jtemp) && J - Jtemp > stepsize * alpha * trace( -Fnt' * gradF)
                break;
            end

            stepsize = beta*stepsize;
            if stepsize < 1.e-16
                % avoid error, keep running
                % error('Extremly small stepsize in polishing step!');
                fprintf('Extremly small stepsize in polishing step!\n');
                stop = 1;
                break;
            end

        end

        % update current step
        F = Ftemp;
        L = Ltemp;
        J = Jtemp;
        
        if stop == 1
            break;
        end
    
    end
    
    if (NTstep == NT_Max_Iter) && (~quiet)
        disp('Maximum number of Newton method reached!')
        disp(['The norm of gradient is ', num2str(ngradF), '.'])
    end    
    
    Fopt = F;
    Jopt = J;
end
