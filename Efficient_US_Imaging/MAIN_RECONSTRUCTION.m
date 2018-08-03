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
clear;
reset(gpuDevice(1));

restoredefaultpath();

addpath('../lib');

run('matlab/vl_setupnn');


rng('shuffle');

%%
epoch       	= 200;
netName         = ['net-epoch-' num2str(epoch) '.mat'];

%%
wgt             = 1e3;



inputSize    	= [64, 384, 1];
wrapSize        = [0, 0, 1];
dsr             = 2;


%%
method_        	= 'normal';
% method_        	= 'residual';
layerFile     	= @cnn_sparse_view_init_multi;

expDir          = ['data/' func2str(layerFile) '_' method_];
addDir          = ['_dsr' num2str(dsr) '_input' num2str(inputSize(1))];

expDirRM4896    = [expDir, addDir];



%%
netRM4896             = load([expDirRM4896, '/' netName]);

gpus                = 2;
mode                = 'test';


dataDir             = [expDirRM4896];


dataFile            = ['DNN4x1_TestVal.mat'];

imdb                = load(fullfile(dataDir, dataFile));
imdb.images.data = single(imdb.images.data);
imdb.images.labels = single(imdb.images.labels);


%%
opts.imageSize      = [size(imdb.images.data,1), size(imdb.images.data,2), size(imdb.images.data,3)];
opts.inputSize      = [size(imdb.images.data,1), size(imdb.images.data,2), size(imdb.images.data,3)];
opts.wrapSize       = wrapSize;

rec                 = zeros(size(imdb.images.data), 'single');
nZ                 = size(imdb.images.data,4);


opt_                = netRM4896.net.meta.trainOpts;
net_                = netRM4896.net;
net_.layers(end)    = [];

            if gpus
                net_     = vl_simplenn_move(net_, 'gpu') ;
            end

for iz = 1:nZ
                
    if gpus
                            data_   = (imdb.images.data(:,:,:,iz) * wgt);
         data_   = gpuArray(data_);
    else
        data_   = (imdb.images.data(:,:,:,iz) * wgt);        
    end

     res_        = vl_simplenn(net_, data_, [], [], 'conserveMemory', 0, 'mode', mode, 'accumulate', 0, ...
                                                    'backPropDepth', inf, 'sync', 0, 'cudnn', 1);

    if strcmp (method_, 'residual')
        err                 = res_(end).x;
        rec_            	= gather((data_ - err) ./ wgt);
        data_             	= gather(data_ ./ wgt);
    else
        rec_                = gather(res_(end).x ./ wgt);
        data_               = gather(data_ ./ wgt);
    end

    clear res_;
  
    labels_     = imdb.images.labels(:,:,:,iz);

    rec(:,:,:,iz)	= rec_;

    disp([num2str(iz) ' / ' num2str(nZ)]);
    
    view = nZ;

    drawnow();
                
end
net = 'rm';
trn = nZ;
view = nZ;
save([expDirRM4896 '/rec_train_view',dataFile], 'rec', '-v7.3');
            

