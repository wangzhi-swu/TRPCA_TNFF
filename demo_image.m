addpath(genpath(cd));
clear;
clc;

pic_name = './pic3.jpg';
X = double(imread(pic_name));                           % Reading Data
    
X = X./255;                                             % Normalization
maxP = max(abs(X(:)));                                  % Maximum Element
[n1,n2,n3] = size(X);                                   % Data Dimensions
X_true = X;
    
sigma = 0.1;
noise = sigma*randn(n1, n2, n3);
maxN = max(noise(:));
Xn = X_true + noise;                                    % Gaussian noise

[n1,n2,n3] = size(Xn);
opts.mu = 1e-4;                                         % parameter miu
opts.tol = 1e-8;                                        % Iteration Termination Condition
opts.rho = 1.1;                                         % parameter rho
opts.max_iter = 500;                                    % max_iter 
opts.DEBUG = 1;

opts.alpha = 0.05;
lambda = 0.001;

PSNR_MAX = 0;
SSIM_MAX = 0;
Time_MAX = 0;
R_MAX = 0;

for R = 1:10
    opts.R = R;
    tic
    [Lhat, Shat, err] = TNFF_trpca_tnn(Xn, lambda, opts);
    time = toc
    psnr = PSNR(X_true, Lhat, maxP)                     % PSNR
    SSIM = ssim(X_true, Lhat)                           % SSIM
    if psnr > PSNR_MAX
        PSNR_MAX = psnr;
        SSIM_MAX = SSIM;
        Time_MAX = time;
        R_MAX = R;
    end
end

PSNR_MAX
SSIM_MAX
Time_MAX
R_MAX
