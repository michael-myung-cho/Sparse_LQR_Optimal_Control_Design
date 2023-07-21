%% F-minimization step

% Solve the F-minimization problem using Anderson-Moore method
% 
% minimize  J(F) + (rho/2) * norm( F - U, 'fro' )^2
%
% Syntax:
% F = Fmin(A,B1,B2,Q,R,U,rho,F0,tol);
%
% Inputs: 
% state-space representation data {A,B1,B2,Q,R},
% matrix U (a linear combination of Lambda and G),
% augmented Lagrangian parameter rho,
% initial condition F0,
% stopping criterion tolerance tol.
%
% Outputs: 
% minimizer F.
%
% More information can be found in the paper linked at:
% http://www.ece.umn.edu/users/fu/papers/ADMM_TAC_2011.pdf

function F = Fmin(A,B1,B2,Q,R,U,rho,F0,tol)
    
    % check if F0 is a stabilizing feedback gain
    maxEigAcl  = max( real( eig( A - B2 * F0 ) ) );
    if maxEigAcl > 0
        error('Initial condition F0 is not stabilizing in Anderson-Moore method!')
    end  

    IR = inv(R);
    F  = F0;    
    AM_Max_Iter = 100;
    

    % control of display messages
    quiet = 1;  
    
    % controllability gramians and objective function phi
      L = lyap( A - B2 * F, B1 * B1' );       
    phi = trace( L * ( Q + F' * R * F ) ) + (rho/2) * norm(F - U, 'fro')^2;
    
    %Anderson-Moore method to solve the F-minimization problem
    for k = 1 : AM_Max_Iter
        
        % observability gramian
        P = lyap( (A - B2 * F)', Q + F' * R * F );

        % one Sylvester equation for F
        Fbar   = lyap(rho * IR, 2 * L, - IR * ( 2 * B2' * P * L + rho * U ));
        
        % descent direction Ftilde
        Ftilde = Fbar - F;

        % gradient direction
        gradphi = 2 * ( R * F - B2' * P ) * L + rho * (F - U);

        % check if Ftilde is a descent direction;    
        if trace( Ftilde' * gradphi ) > 1.e-10
            error('Ftilde is not a descent direction!')
        end 

        if norm( gradphi, 'fro' ) < tol
            break;
        end

        stepsize = 1;

        while 1

            Ftemp = F + stepsize * Ftilde;
            maxEigAcltemp = max( real( eig( A - B2 * Ftemp ) ) );

            % the objective function is infinity if Ftemp is not stabilizing
            if maxEigAcltemp > 0 
                phitemp = nan;
                
            % otherwise compute the closed-loop H2 norm 
            else
                Ltemp   = lyap( A - B2 * Ftemp, B1 * B1' );
                phitemp = trace( Ltemp * (Q + Ftemp' * R * Ftemp) ) + ...
                    (rho/2) * norm( Ftemp - U, 'fro' )^2;       
            end

            % Armijo rule data
            alpha = 0.3; 
            beta  = 0.5;
            if ~isnan(phitemp) && phi - phitemp > ...
                    stepsize * alpha * trace( - Ftilde' * gradphi )
                break;
            end                
            stepsize = stepsize * beta;

            if stepsize < 1.e-16            
                error('Extremely small stepsize in F-minimization step!');            
            end        
        end 

        % update current step
          F = Ftemp;    
          L = Ltemp;
        phi = phitemp;
    end
    %fprintf('Number of iterations in AM: %d\n',k);
    
    if (k == AM_Max_Iter) && (~quiet)
        disp('Maximum number of Anderson-Moore method reached!')
        disp(['The norm of gradient is ', num2str( norm( gradphi, 'fro' ) ), '.'])
    end

end
