%% Conjugate gradient method to compute newton direction

% Inputs: 
% state-space representation data {A,B2,F,P,L,R},
% gradient gradF, 
% structural identity IS, 
% initial condition Fnt0.
%
% Output: 
% Newton direction Fnt.
%
% More information can be found in the paper linked at:
% http://www.ece.umn.edu/users/fu/papers/ADMM_TAC_2011.pdf

function Fnt = newtonCG(A,B2,F,P,L,R,gradF,IS)

    % initialization
    % number of nonzero elements
         q = nnz(IS);    
        Pi = gradF;
     Delta = - gradF;
    Ftilde = zeros(size(IS));
    ngradF = norm( gradF, 'fro' );
    
    % closed-loop A-matrix
    Acl = A - B2 * F;
    % denote Z  = R * F - B2' * P 
    % to save computation
      Z = R * F - B2' * P;

    % control of display messages
    quiet = 1;    
    
    % conjugate gradient scheme
    for k = 0 : q
        
        % compute H
        % Z = R * F - B2' * P;
        G1 = B2 * Delta * L;
        G2 = -Z' * Delta;
        Ltilde = lyap( Acl, - G1 - G1' );
        Ptilde = lyap( Acl', - G2 - G2');
        H = 2 * ( ( ( R * Delta - B2' * Ptilde ) * L + Z * Ltilde ) .* IS );        
        
        % negative curvature test  
        % form the inner product between H and Delta
        trHDelta = sum( sum( H .* Delta ) );
        if ( trHDelta <= 0) && (k == 0)
            Fnt = - gradF;
            if ~quiet
                disp('Negative curvature detected!')
            end
            break;
        elseif ( trHDelta <= 0) && (k > 0)
            Fnt = Ftilde;
            if ~quiet
                disp('Negative curvature detected!')
            end
            break;
        end
        
        alpha  = - sum( sum( Pi .* Delta ) ) / trHDelta;
        Ftilde = Ftilde + alpha * Delta;
        Pi     = Pi + alpha * H;
        beta   = sum( sum(  H .* Pi ) ) / trHDelta;
        Delta  = - Pi + beta * Delta;
                
        % Nocedal and Wright '99 p140
        % Nocedal and Wright '06 p168
        % stopping criterion for the conjugate gradient method
        if norm( Pi, 'fro' ) < min( 0.5, sqrt( ngradF ) ) * ngradF
            Fnt = Ftilde;
            break;
        end
    end
    
    if k == q
        Fnt = Ftilde;
        if ~quiet
            disp('Maximum number of conjugate gradient method reached!')
            disp(['Norm of gradient is ',num2str( ngradF, '%10.2E' ),'.'])
        end            
    end
        
end
