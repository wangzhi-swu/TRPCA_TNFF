function [X] = TNFF_solver(C,rho,a,rr)
    [U, S, V] = svd(C,'econ');                % perform SVD on matrix C
    [X, T1] = solve_P(S,rho,a,U,V, rr);
end

function [X,S1] = solve_P(S,rho,a,U,V, rr)
    tau = 1/rho;                              % tau = 1/rho
    S0 = diag(S);                             % the diagonal matrix of S: S0 = σ(C)
    m = length(S0);                           % number of elements in matrix σ(C): m = σ(C)
    S1 = zeros(m,1);                          % S1：initialize a column vector of the same length
    if tau <= 1/(2*a^2)
        t = (tau*a);
    else
        t = max(sqrt(2*tau)-1/(2*a),0);
    end

    S1 = S0;
    rr = rr+1;
    for i= rr:m
            if S0(i) > t 
                S1(i) = prox(tau,a,S0(i));
            else
                S1(i) = 0;
            end
    end
    X = U *diag(S1) *V';
end

function [s1] = prox(tau,a,sigma)
    p1 = (1+a*abs(sigma))/3;
    f = phi(tau,a,sigma);
    p2 = 1+2*cos(f/3 - pi/3);
    s1 = sign(sigma)*((p1*p2 - 1)/a);
end

function [f] = phi(tau,a,sigma)
    num = 27 * tau * (a^2);
    den = 2 * (1 + a*abs(sigma))^3;
    f = acos(num/den - 1);
end
   

