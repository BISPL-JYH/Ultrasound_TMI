%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by 'Shujaat Khan' (shujaat@kaist.ac.kr) at 2018.8.2
% IEEE-TMI Paper : Efficient B-mode Ultrasound Image Reconstruction from Sub-sampled RF Data using Deep Learning


% Copyright <2018> <Shujaat Khan' (shujaat@kaist.ac.kr)>
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

clc
clear;
reset(gpuDevice);


rng('shuffle');


dataDir             = ['data'];


dataFile            = ['DNN4x1_TrVal.mat'];
imdb                = load(fullfile(dataDir, dataFile));

dsr                 = 2;

inputRange          = [-1, 1];
inputSize           = [64, 384, 1];

wrapSize            = [0, 0, 1];

numEpochs           = 400;

batchSize           = 1;
numSubBatches       = 1;

cwgt                = 1e0;
grdclp              = [-1e-2, 1e-2];
lrnrate             = [-7, -9]; 
wgtdecay            = 1e-4;

wgt                 = 1e3;
offset              = 0;

smp                 = 1;

% CNN Parameters
batchNormalization  = true;
gpus                = 2;

% return ;
%
method              = 'normal';
% method         	= 'residual';
% strFunc         = ['cnn_sparse_view_init_C'];
% strFunc         = ['cnn_sparse_view_init_R'];
strFunc         = ['cnn_sparse_view_init_multi'];
% strFunc         = ['cnn_sparse_view_init_single'];

layerFile       = str2func(strFunc);

expDir          = ['data/'];
expDir          = [expDir func2str(layerFile) '_' method];
addDir          = ['_dsr' num2str(dsr) '_input' num2str(inputSize(1))];

expDir          = [expDir, addDir];

train   	= struct('gpus', gpus);

%%
[net_train, info_train]         = cnn_sparse_view_train( ...
    'method',       method,         'smp',                  smp,   ...
    'imdb',         imdb,           'train',                train, ...
    'cwgt',         cwgt,           'grdclp',               grdclp,                 'lrnrate',              lrnrate,                ...
    'wgtdecay',     wgtdecay,    	'expDir',               expDir,                 'dataDir',              dataDir,                ...
    'dataFile',     dataFile,       'layerFile',            layerFile,              'batchNormalization',	batchNormalization,  	...
    'dsr',          dsr,            'inputRange',           inputRange,             'inputSize',            inputSize,              ...
    'wrapSize',     wrapSize,       'wgt',                  wgt,                    'offset',               offset,                 ...
    'numEpochs',    numEpochs,      'numSubBatches',        numSubBatches,          'batchSize',            batchSize);

return ;
