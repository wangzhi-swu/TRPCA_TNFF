function [L,S, res] = TNFF_trpca_tnn(X,lambda,opts)
%
% min_{L,S} ||L||_*+lambda*||S||_1, s.t. X=L+S
% ---------------------------------------------
% Input:
%       X       -    d1*d2*d3 tensor
%       lambda  -    > 0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance                         
%           opts.max_iter   -   maximum number of iterations                  
%           opts.mu         -   stepsize for dual variable updating in ADMM   
%           opts.max_mu     -   maximum stepsize                              
%           opts.rho        -   rho>=1, ratio used to increase mu             
%           opts.DEBUG      -   0 or 1
%
% Output:
%       L       -    d1*d2*d3 tensor
%       S       -    d1*d2*d3 tensor
%       obj     -    objective function value
%       err     -    residual 
%       iter    -    number of iterations
% ---------------------------------------------

tol = 1e-8; 
max_iter = 500;
rho = 1.1;
alpha = 0.2;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

if isfield(opts, 'alpha');       alpha = opts.alpha;                end
if isfield(opts, 'R');       R = opts.R;                end



dim = size(X)
[n1,n2,n3] = size(X);
L = zeros(dim); 
S = L;
Y = L;

iter = 0;
for iter = 1 : max_iter
    Lk = L;
    Sk = S;

%     % update L —— min_X rho*||X||_*+0.5*||X-Y||_F^2
%     [L,tnnL] = prox_tnn(-S+X-Y/mu,1/mu);

%% update L —— min_X Pa(X) + rho||X-Y||_F^2
    L_tmp = -S+X-Y/mu;
    L_tmp = fft(L_tmp,[],3);
    for i = 1 : n3
        [L(:,:,i)] = TNFF_solver(L_tmp(:,:,i), mu, alpha, R);
    end
    L = ifft(L,[],3);


    %% update S —— min_x lambda*||x||_1+0.5*||x-b||_2^2
    S = prox_l1(-L+X-Y/mu,lambda/mu);
  
    dY = L+S-X;
    chgL = max(abs(Lk(:)-L(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chgX = max(abs(dY(:)));
    chg = max([ chgL chgS max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
%             obj = tnnL+lambda*norm(S(:),1);
            err = norm(dY(:));
%             disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
%                     ', obj=' num2str(obj) ', err=' num2str(err)]); 
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);    

    res.ierr(iter) = norm(dY(:))/norm(X(:));
    res.iL(iter) = norm(L(:)-Lk(:))/norm(L(:));
    res.iS(iter) = norm(S(:)-Sk(:))/norm(S(:));
end
obj = 0;
% obj = tnnL+lambda*norm(S(:),1);
err = norm(dY(:));
