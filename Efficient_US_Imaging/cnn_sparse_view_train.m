function [net, info] = cnn_sparse_view_train(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')), 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.method         = 'normal';

opts.dsr            = 2;
opts.inputRange     = [0, 1];
opts.inputSize      = [50, 50, 1];
opts.wrapSize       = [1, 1, 1];

opts.cwgt           = 1e-3;
opts.grdclp         = [-1e-2, 1e-2];
opts.lrnrate     	= [-3, -4];
opts.wgtdecay     	= [-3, -4];

opts.lv             = [];
opts.pflt           = 'pyr';
opts.dflt           = 'cd';

opts.wgt            = 1;
opts.offset         = 0;

opts.numEpochs      = 300;
opts.batchSize      = 10;
opts.numSubBatches  = 1;

opts.imdb           = [];
opts.smp            = 0.5;

[opts, varargin]    = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.dataFile = 'imdb.mat';
opts.layerFile  = 'cnn_denoise_init';
opts.train = struct() ;
[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.imdbPath = fullfile(opts.dataDir, opts.dataFile);
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
net     = opts.layerFile( ...
    'batchNormalization',	opts.batchNormalization,	'networkType',      opts.networkType,	...
    'dsr',                  opts.dsr,                   'inputRange',       opts.inputRange,    ...
    'inputSize',            opts.inputSize,             'wrapSize',         opts.wrapSize,      ...
    'wgt',                  opts.wgt,                   'offset',           opts.offset,        ...
    'cwgt',                 opts.cwgt,                  'grdclp',           opts.grdclp,        ...
    'lrnrate',              opts.lrnrate,               'wgtdecay',         opts.wgtdecay,      ...
    'numEpochs',            opts.numEpochs,             'method',           opts.method,        ...
    'batchSize',            opts.batchSize,             'numSubBatches',	opts.numSubBatches, ...
    'lv',                   opts.lv,                    ...
    'pflt',                 opts.pflt,                  'dflt',             opts.dflt) ;

if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end

if exist(opts.imdbPath, 'file')
    %   imdb = load(opts.imdbPath) ;
    imdb	= opts.imdb;
else
    imdb = getMnistImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainfn  = @cnn_train ;
    case 'dagnn', trainfn     = @cnn_train_dag ;
end

% edited by ys
% valset  = find(imdb.images.set == 3);
% valset  = valset(1:8:end);

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, 'smp', opts.smp, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
%   'val', valset) ;


% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        %     fn = @(x,y) getSimpleNNBatch(x,y) ;
        fn = @(x,y) getSimpleNNBatch(x,y,opts) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% % --------------------------------------------------------------------
% function [images, labels] = getSimpleNNBatch(imdb, batch)
% % --------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
% function [images, labels, dcs] = getSimpleNNBatch(imdb, batch, patch, wgt, offset)
function [data, labels, orig] = getSimpleNNBatch(imdb, batch, opts)
% function [data, labels, data_mean] = getSimpleNNBatch(imdb, batch, opts)
% --------------------------------------------------------------------
rng('shuffle');

patch   = opts.inputSize;
wgt     = opts.wgt;
offset  = opts.offset;
lv  	= opts.lv;
pflt    = opts.pflt;
dflt    = opts.dflt;

ny      = size(imdb.images.labels, 1);
nx      = size(imdb.images.labels, 2);
nch     = sum([1; 2.^lv(:)]);
nz      = length(batch);


iy      = floor(rand(1)*(ny - patch(1)));
ix      = floor(rand(1)*(nx - patch(2)));

by      = 1:patch(1);
bx      = 1:patch(2);

data_ 	= wgt*single(imdb.images.data(iy + by,ix + bx,:,batch)) + offset ;
labels_	= wgt*single(imdb.images.labels(iy + by,ix + bx,:,batch)) + offset ;

% data    = zeros([patch(1), patch(2), nch, nz], 'single');
% labels  = zeros([patch(1), patch(2), nch, nz], 'single');
% 
% if ~isempty(lv)
%     
%     for iz = 1:nz
%         
%         % Nonsubsampled Contourlet decomposition
%         data(:,:,:,iz)      = wavecell2mat(nsctdec( double(data_(:,:,:,iz)), dflt, pflt, lv ), lv);
%         labels(:,:,:,iz)	= wavecell2mat(nsctdec( double(labels_(:,:,:,iz)), dflt, pflt, lv ), lv);
% 
% %         for ich = 1:nch
% %             figure(100); colormap gray;
% %             subplot(131);   imagesc(data(:,:,ich,iz));
% %             subplot(132);   imagesc(labels(:,:,ich,iz));
% %             subplot(133);   imagesc(data(:,:,ich,iz) - labels(:,:,ich,iz));
% %             pause(0.1);
% % 
% % %             drawnow();
% %         end
%         
%     end
%     
% else
%     data    = data_;
%     labels  = labels_;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CASE 1
data    = data_;
labels  = labels_;
orig    = labels_;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CASE 2
% data        = data_;
% labels      = labels_;
% data_mean   = mean(mean(mean(data, 1), 2), 3);
% 
% data        = bsxfun(@minus, data, data_mean);
% labels      = bsxfun(@minus, labels, data_mean);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (imdb.images.set(1) == 1)
    if (rand > 0.5)
        data    = flip(data, 1);
        labels  = flip(labels, 1);
        orig    = flip(orig, 1);
    end
    
    if (rand > 0.5)
        data    = flip(data, 2);
        labels  = flip(labels, 2);
        orig    = flip(orig, 2);
    end
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte', ...
    't10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir) ;
end

for i=1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, opts.dataDir) ;
    end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
