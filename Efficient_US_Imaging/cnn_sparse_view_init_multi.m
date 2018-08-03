function net = cnn_sparse_view_init_multi(varargin)
% CNN_DENOISE_LENET Initialize a CNN similar for DENOISE

% rng('default');
% rng(0) ;
rng('shuffle');

opts.batchNormalization	= true ;
opts.networkType        = 'simplenn' ;

opts.method             = 'normal';

% nch                     = 1;
opts.ext                = 1;
opts.dsr                = 2;

opts.lv                 = [];
opts.pflt               = 'pyr';
opts.dflt               = 'cd';

opts.wgt                = 1;
opts.offset             = 0;

opts.numEpochs          = 300;
opts.batchSize          = 10;
opts.numSubBatches      = 1;

opts.inputRange         = [0, 1];

[opts, varargin]        = vl_argparse(opts, varargin) ;

nch                     = 1;%sum([1; 2.^opts.lv(:)]);

opts.inputSize          = [50, 50, nch];
opts.wrapSize           = [1, 1, 1];

opts.cwgt               = 1e-3;
opts.grdclp             = [-1e-2, 1e-2];
opts.lrnrate            = [-3, -4];
opts.wgtdecay           = 1e-4;

opts                    = vl_argparse(opts, varargin) ;

% Meta parameters
net.meta.inputSize                  = opts.inputSize ;

net.meta.trainOpts.errorFunction	= 'euclidean';

net.meta.trainOpts.numEpochs        = opts.numEpochs ;
net.meta.trainOpts.batchSize        = opts.batchSize ;
net.meta.trainOpts.numSubBatches    = opts.numSubBatches ;

net.meta.trainOpts.learningRate     = logspace(opts.lrnrate(1), opts.lrnrate(2), floor(opts.numEpochs)) ;

net.meta.trainOpts.weightDecay      = opts.wgtdecay;
net.meta.trainOpts.momentum         = 9e-1;

net.meta.trainOpts.gradMin          = opts.grdclp(1);
net.meta.trainOpts.gradMax          = opts.grdclp(2);

net.meta.trainOpts.method       	= opts.method;

net.meta.trainOpts.inputRange    	= opts.inputRange;

net.meta.trainOpts.lv               = opts.lv;
net.meta.trainOpts.pflt             = opts.pflt;
net.meta.trainOpts.dflt             = opts.dflt;

net.meta.trainOpts.wgt          	= opts.wgt;
net.meta.trainOpts.offset       	= opts.offset;


net.layers                    	= {};

%
ext                             = opts.ext;
cf                            	= opts.cwgt ;    % 1e-1 > f > 1e-3 

bf                              = 1;
bmu                             = 0;
bsgm                            = 0;

flt1                            = 3;
hflt1                           = floor(flt1/2);
out1                            = 64*ext;

flt2                            = 3;
hflt2                           = floor(flt2/2);
out2                            = 128*ext;

flt3                            = 3;
hflt3                           = floor(flt3/2);
out3                            = 256*ext;

flt4                            = 3;
hflt4                           = floor(flt4/2);
out4                            = 512*ext;

flt5                            = 3;
hflt5                           = floor(flt5/2);
out5                            = 1024*ext;

flte                            = 1;
hflte                           = floor(flte/2);
oute                            = 1;%1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 1.	512 * 512
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% net  	= addLayer('wavedec', 'lv', lv, 'pflt', pflt, 'dflt', dflt, 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, nch], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('send', 'send', [1], 'net', net);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 2.	512 * 512 -> 256 * 256
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('pool', 'method', 'max', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out1], 'output', out2, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out2], 'output', out2, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out2], 'output', out2, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('send', 'send', [2], 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 3.	256 * 256 -> 128 * 128
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('pool', 'method', 'max', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out2], 'output', out3, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out3], 'output', out3, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out3], 'output', out3, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('send', 'send', [3], 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 4.	128 * 128 -> 64 * 64
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('pool', 'method', 'max', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out3], 'output', out4, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out4], 'output', out4, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net   	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out4], 'output', out4, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('send', 'send', [4], 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 5.	64 * 64 -> 32 * 32
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('pool', 'method', 'max', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt5, flt5, out4], 'output', out5, 'stride', [1, 1], 'pad', hflt5*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt5, flt5, out5], 'output', out5, 'stride', [1, 1], 'pad', hflt5*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt5, flt5, out5], 'output', out4, 'stride', [1, 1], 'pad', hflt5*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 6.    32 * 32 -> 64 * 64
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('unpool', 'method', 'avg', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net, 'ipool', 0);
% net     = addLayer('concat', 'concat', [4], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out5], 'output', out4, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net   	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out4], 'output', out4, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net   	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt4, flt4, out4], 'output', out3, 'stride', [1, 1], 'pad', hflt4*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 7.    64 * 64 -> 128 * 128
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('unpool', 'method', 'avg', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net, 'ipool', 0);
% net     = addLayer('concat', 'concat', [3], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out4], 'output', out3, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out3], 'output', out3, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net 	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt3, flt3, out3], 'output', out2, 'stride', [1, 1], 'pad', hflt3*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 8.    128* 128 -> 256 * 256
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('unpool', 'method', 'avg', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net, 'ipool', 0);
% net     = addLayer('concat', 'concat', [2], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out3], 'output', out2, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out2], 'output', out2, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt2, flt2, out2], 'output', out1, 'stride', [1, 1], 'pad', hflt2*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % BLOCK # 9.    256 * 256 -> 512 * 512
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% net     = addLayer('unpool', 'method', 'avg', 'pool', [2, 2], 'stride', [2, 2], 'pad', [0, 0, 0, 0], 'net', net, 'ipool', 0);
% net     = addLayer('concat', 'concat', [1], 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt1, flt1, out2], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
% net  	= addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
% net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
% net     = addLayer('relu', 'net', net);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 2.	512 * 512 -> 256 * 256
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('send', 'send', [2], 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 3.	256 * 256 -> 128 * 128
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('send', 'send', [3], 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 4.	128 * 128 -> 64 * 64
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('send', 'send', [4], 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 5.	64 * 64 -> 32 * 32
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 6.    32 * 32 -> 64 * 64
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('concat', 'concat', [4], 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, 2*out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 7.    64 * 64 -> 128 * 128
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('concat', 'concat', [3], 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, 2*out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 8.    128* 128 -> 256 * 256
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('concat', 'concat', [2], 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, 2*out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 9.    256 * 256 -> 512 * 512
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('concat', 'concat', [1], 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, 2*out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

net     = addLayer('conv', 'filter', [flt1, flt1, out1], 'output', out1, 'stride', [1, 1], 'pad', hflt1*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('bnorm', 'f', bf, 'mu', bmu, 'sgm', bsgm, 'opts', opts, 'net', net);
net     = addLayer('relu', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% net76}  = addLayer('concat', 'concat', [4, 16, 28, 40, 52, 64], 'net', net);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BLOCK # 10.    512 * 512 -> 512 * 512
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net     = addLayer('conv', 'filter', [flte, flte, out1], 'output', oute, 'stride', [1, 1], 'pad', hflte*ones(4, 1), 'f', cf, 'net', net);
net     = addLayer('euclideanloss', 'net', net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% optionally switch to batch normalization
% if opts.batchNormalization
%   net = insertBnorm(net, 1) ;
%   net = insertBnorm(net, 4) ;
%   net = insertBnorm(net, 7) ;
% end

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;


% --------------------------------------------------------------------
function net = addBnorm(net)
% --------------------------------------------------------------------
assert(isfield(net.layers{end}, 'weights'));
ndim = size(net.layers{end}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{end}.biases = [] ;
net.layers = horzcat(net.layers, layer) ;