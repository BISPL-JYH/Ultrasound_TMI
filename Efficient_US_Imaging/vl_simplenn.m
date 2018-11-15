function res = vl_simplenn(net, x, dzdy, res, varargin)
%VL_SIMPLENN  Evaluate a SimpleNN network.
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY (foward+bacwkard pass).
%   RES = VL_SIMPLENN(NET, X, [], RES) evaluates the NET on X reusing the
%   structure RES.
%   RES = VL_SIMPLENN(NET, X, DZDY, RES) evaluates the NET on X and its
%   derivatives reusing the structure RES.
%
%   This function process networks using the SimpleNN wrapper
%   format. Such networks are 'simple' in the sense that they consist
%   of a linear sequence of computational layers. You can use the
%   `dagnn.DagNN` wrapper for more complex topologies, or write your
%   own wrapper around MatConvNet computational blocks for even
%   greater flexibility.
%
%   The format of the network structure NET and of the result
%   structure RES are described in some detail below. Most networks
%   expect the input data X to be standardized, for example by
%   rescaling the input image(s) and subtracting a mean. Doing so is
%   left to the user, but information on how to do this is usually
%   contained in the `net.meta` field of the NET structure (see
%   below).
%
%   The NET structure needs to be updated as new features are
%   introduced in MatConvNet; use the `VL_SIMPLENN_TIDY()` function
%   to make an old network current, as well as to cleanup and check
%   the structure of an existing network.
%
%   Networks can run either on the CPU or GPU. Use VL_SIMPLENN_MOVE()
%   to move the network parameters between these devices.
%
%   To print or obtain summary of the network structure, use the
%   VL_SIMPLENN_DISPLAY() function.
%
%   VL_SIMPLENN(NET, X, DZDY, RES, 'OPT', VAL, ...) takes the following
%   options:
%
%   `Mode`:: `normal`
%      Specifies the mode of operation. It can be either `'normal'` or
%      `'test'`. In test mode, dropout and batch-normalization are
%      bypassed. Note that, when a network is deployed, it may be
%      preferable to *remove* such blocks altogether.
%
%   `ConserveMemory`:: `false`
%      Aggressively delete intermediate results. This in practice has
%      a very small performance hit and allows training much larger
%      models. However, it can be useful to disable it for
%      debugging. Keeps the values in `res(1)` (input) and `res(end)`
%      (output) with the outputs of `loss` and `softmaxloss` layers.
%      It is also possible to preserve individual layer outputs
%      by setting `net.layers{...}.precious` to `true`.
%      For back-propagation, keeps only the derivatives with respect to
%      weights.
%
%   `CuDNN`:: `true`
%      Use CuDNN when available.
%
%   `Accumulate`:: `false`
%      Accumulate gradients in back-propagation instead of rewriting
%      them. This is useful to break the computation in sub-batches.
%      The gradients are accumulated to the provided RES structure
%      (i.e. to call VL_SIMPLENN(NET, X, DZDY, RES, ...).
%
%   `BackPropDepth`:: `inf`
%      Limit the back-propagation to top-N layers.
%
%   `SkipForward`:: `false`
%      Reuse the output values from the provided RES structure and compute
%      only the derivatives (backward pass).
%
%   ## The result format
%
%   SimpleNN returns the result of its calculations in the RES
%   structure array. RES(1) contains the input to the network, while
%   RES(2), RES(3), ... contain the output of each layer, from first
%   to last. Each entry has the following fields:
%
%   - `res(i+1).x`: the output of layer `i`. Hence `res(1).x` is the
%     network input.
%
%   - `res(i+1).aux`: any auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - `res(i+1).dzdx`: the derivative of the network output relative
%     to the output of layer `i`. In particular `res(1).dzdx` is the
%     derivative of the network output with respect to the network
%     input.
%
%   - `res(i+1).dzdw`: a cell array containing the derivatives of the
%     network output relative to the parameters of layer `i`. It can
%     be a cell array for multiple parameters.
%
%   ## The network format
%
%   The network is represented by the NET structure, which contains
%   two fields:
%
%   - `net.layers` is a cell array with the CNN layers.
%
%   - `net.meta` is a grab-bag of auxiliary application-dependent
%     information, including for example details on how to normalize
%     input data, the class names for a classifiers, or details of
%     the learning algorithm. The content of this field is ignored by
%     VL_SIMPLENN().
%
%   SimpleNN is aware of the following layers:
%
%   Convolution layer::
%     The convolution layer wraps VL_NNCONV(). It has fields:
%
%     - `layer.type` contains the string `'conv'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.stride` is the sampling stride (e.g. 1).
%     - `layer.pad` is the padding (e.g. 0).
%
%   Convolution transpose layer::
%     The convolution transpose layer wraps VL_NNCONVT(). It has fields:
%
%     - `layer.type` contains the string `'convt'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.upsample` is the upsampling factor (e.g. 1).
%     - `layer.crop` is the amount of output cropping (e.g. 0).
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - `layer.type` contains the string `'pool'`.
%     - `layer.method` is the pooling method (either 'max' or 'avg').
%     - `layer.pool` is the pooling size (e.g. 3).
%     - `layer.stride` is the sampling stride (usually 1).
%     - `layer.pad` is the padding (usually 0).
%
%   Normalization (LRN) layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields:
%
%     - `layer.type` contains the string `'normalize'` or `'lrn'`.
%     - `layer.param` contains the normalization parameters (see VL_NNNORMALIZE()).
%
%   Spatial normalization layer::
%     The spatial normalization layer wraps VL_NNSPNORM(). It has fields:
%
%     - `layer.type` contains the string `'spnorm'`.
%     - `layer.param` contains the normalization parameters (see VL_NNSPNORM()).
%
%   Batch normalization layer::
%     This layer wraps VL_NNBNORM(). It has fields:
%
%     - `layer.type` contains the string `'bnorm'`.
%     - `layer.weights` contains is a cell-array with, multiplier and
%       biases, and moments parameters
%
%     Note that moments are used only in `'test'` mode to bypass batch
%     normalization.
%
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - `layer.type` contains the string `'relu'`.
%     - `layer.leak` is the leak factor (e.g. 0).
%
%     The sigmoid layer is the same, but for the sigmoid function,
%     with `relu` replaced by `sigmoid` and no leak factor.
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - `layer.type` contains the string `'dropout'`.
%     - `layer.rate` is the dropout rate (e.g. 0.5).
%
%     Note that the block is bypassed in `test` mode.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - `layer.type` contains the string`'softmax'`.
%
%   Log-loss layer and softmax-log-loss::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - `layer.type` contains `'loss'`.
%     - `layer.class` contains the ground-truth class labels.
%
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS() instead. it
%     has the same parameters, but `type` contains the `'softmaxloss'`
%     string.
%
%   P-dist layer::
%     The p-dist layer wraps VL_NNPDIST(). It has fields:
%
%     - `layer.type` contains the string  `'pdist'`.
%     - `layer.p` is the P parameter of the P-distance (e.g. 2).
%     - `layer.noRoot` it tells whether to raise the distance to
%     the P-th power (e.g. `false`).
%     - `layer.epsilon` is the regularization parameter for the derivatives.
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - `layer.type` contains the string `'custom'`.
%     - `layer.forward` is  a function handle computing the block.
%     - `layer.backward` is a function handle computing the block derivative.
%
%     The first function is called as
%
%          res(i+1) = layer.forward(layer, res(i), res(i+1))
%
%     where RES is the structure array specified before. The second function is
%     called as
%
%          res(i) = layer.backward(layer, res(i), res(i+1))
%
%     Note that the `layer` structure can contain additional custom
%     fields if needed.
%
%   See also: dagnn.DagNN, VL_SIMPLENN_TIDY(),
%   VL_SIMPLENN_DISPLAY(), VL_SIMPLENN_MOVE().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.conserveMemory = true ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  if opts.skipForward
    error('simplenn:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('simplenn:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
  res(1).x = x ;
end

% -------------------------------------------------------------------------
%                                                              By pass
% -------------------------------------------------------------------------
send    = cell(n, 1);
receive = cell(n, 1);
concat  = cell(n, 1);

% send    = cell(0, 0);
% receive = cell(0, 0);
% concat  = cell(0, 0);

for j = 1:n
    l_ = net.layers{j};
    if (strcmp(l_.type, 'send'))
        for k = 1:length(l_.send)
            send{l_.send(k)} = [send{l_.send(k)} j];
        end
    elseif (strcmp(l_.type, 'receive'))
        for k = 1:length(l_.receive)
            receive{l_.receive(k)} = [receive{l_.receive(k)} j];
        end
    elseif (strcmp(l_.type, 'concat'))
        for k = 1:length(l_.concat)
            concat{l_.concat(k)} = [concat{l_.concat(k)} j];
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
%     i
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;

    case 'convt'
      res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
        'crop', l.crop, ...
        'upsample', l.upsample, ...
        'numGroups', l.numGroups, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
    case 'subpixel'
        res(i+1).x = vl_nnsubpixel(res(i).x, l.dy, l.dx);
        
    case 'subpixelt'
        res(i+1).x = vl_nnsubpixelt(res(i).x, l.dy, l.dx);
            
    case 'wavedec'
        res(i+1).x = vl_nnwavedec(res(i).x, l.lv, l.pflt, l.dflt);
        
    case 'waverec'
        res(i+1).x = vl_nnwaverec(res(i).x, l.lv, l.pflt, l.dflt);

    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;

    case 'unpool'
        if (l.ipool > 0)
            res(i+1).x = vl_nnpool(res(l.ipool).x, l.pool, res(i).x, ...
                'pad', l.pad, 'stride', l.stride, ...
                'method', l.method, ...
                l.opts{:}, ...
                cudnn{:}) ;
        else
            sz      = size(res(i).x);
            
            sz_     = sz;
            sz_(1)	= sz(1)*l.stride(1);
            sz_(2)  = sz(2)*l.stride(2);
            
            x_    	= zeros(sz_, 'like', res(i).x);
            
            switch l.method
                case 'max'
                    if (l.ipool == 0)
                        %             Version 1 : UNIFORM GRID
                        x_(1:l.stride(1):end, 1:l.stride(2):end, :, :)	= 1;
                    elseif (l.ipool < 0)
                        
                        %             Version 2 : RANDOM GRID
                        idy  	= repmat((1:l.stride(1):sz_(1))',               [1, sz(2)*sz(3)*sz(4)]) ;
                        idx   	= repmat(1:l.stride(2):sz_(2)*sz(3)*sz(4),      [sz(1), 1]) ;
                        
                        idy_    = zeros(sz(1), sz(2), sz(3), sz(4), 'single');
                        idx_    = zeros(sz(1), sz(2), sz(3), sz(4), 'single');
                        
                        for ich = 1:sz(4)
                            idy_(:,:,:, ich)    = repmat(fix(rand(sz(1), sz(2))*l.pool(1)), [1, 1, sz(3)]);
                            idx_(:,:,:, ich)    = repmat(fix(rand(sz(1), sz(2))*l.pool(2)), [1, 1, sz(3)]);
                        end
                        
                        ind   	= sub2ind([sz_(1), sz_(2)*sz_(3)*sz_(4)], idy(:) + idy_(:), idx(:) + idx_(:));
                        x_(ind)	= 1;
                    end
                case 'avg'
                    x_(:)   = 1;
            end
            
            res(i+1).x = vl_nnpool(x_, l.pool, res(i).x, ...
                'pad', l.pad, 'stride', l.stride, ...
                'method', l.method, ...
                l.opts{:}, ...
                cudnn{:}) ;
            
            clear x_;
        end
        
    case {'normalize', 'lrn'}
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;

    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;

    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;

    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
      
    case 'euclideanloss'
      res(i+1).x = vl_nneuclideanloss(res(i).x, l.class) ;

    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;

    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;

    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;

    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;

    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end

    case 'bnorm'
%       if testMode
%         res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
%       else
%         res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
%       end
      res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;

    case 'pdist'
      res(i+1).x = vl_nnpdist(res(i).x, l.class, l.p, ...
        'noRoot', l.noRoot, ...
        'epsilon', l.epsilon, ...
        'aggregate', l.aggregate, ...
        'instanceWeights', l.instanceWeights) ;
    
    case 'send'
      res(i+1).x = res(i).x;

    case 'receive'
      res(i+1).x = res(i).x;
      
      for j = [send{l.receive}]
          res(i+1).x = res(i+1).x + res(j).x;
      end
      
    case 'concat'
      res(i+1).x = cat(3, res([send{l.concat}, i]).x);

    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;

    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end

  % optionally forget intermediate results
  needsBProp = doder && i >= backPropLim;
  forget = opts.conserveMemory && ~needsBProp ;
  if i > 1
    lp = net.layers{i-1} ;
    % forget RELU input, even for BPROP
    forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
    forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss') || strcmp(lp.type, 'euclideanloss')) ;
    forget = forget && ~lp.precious ;
  end
  if forget
    res(i).x = [] ;
  end
%   if opts.conserveMemory && isempty(find(cell2mat(send) == i, 1)) && i > 1 && ~strcmp(l.type, 'relu')
%     res(i).x = [] ;
%   end

  if gpuMode && opts.sync
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:backPropLim
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type

      case 'conv'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'pad', l.pad, ...
          'stride', l.stride, ...
          l.opts{:}, ...
          cudnn{:}) ;

      case 'convt'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'crop', l.crop, ...
          'upsample', l.upsample, ...
          'numGroups', l.numGroups, ...
          l.opts{:}, ...
          cudnn{:}) ;
      
      case 'subpixel'
        res(i).dzdx     = vl_nnsubpixel(res(i).x, l.dy, l.dx, res(i+1).dzdx);
        
      case 'subpixelt'
        res(i).dzdx     = vl_nnsubpixelt(res(i).x, l.dy, l.dx, res(i+1).dzdx);
        
    case 'wavedec'
        res(i+1).dzdx	= vl_nnwavedec(res(i).x, l.lv, l.pflt, l.dflt, res(i+1).dzdx);
        
    case 'waverec'
        res(i+1).dzdx	= vl_nnwaverec(res(i).x, l.lv, l.pflt, l.dflt, res(i+1).dzdx);

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, ...
                                cudnn{:}) ;
       
      case 'unpool'
        res(i).dzdx = vl_nnpool(res(i+1).dzdx, l.pool, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, ...
                                cudnn{:}) ;

      case {'normalize', 'lrn'}
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;

      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;

      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        
      case 'euclideanloss'
        res(i).dzdx = vl_nneuclideanloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end

      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;

      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;

      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;

      case 'dropout'
        if testMode
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end

      case 'bnorm'
        [res(i).dzdx, dzdw{1}, dzdw{2}, dzdw{3}] = ...
          vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx) ;
        % multiply the moments update by the number of images in the batch
        % this is required to make the update additive for subbatches
        % and will eventually be normalized away
        dzdw{3} = dzdw{3} * size(res(i).x,4) ;

      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.class, ...
          l.p, res(i+1).dzdx, ...
          'noRoot', l.noRoot, ...
          'epsilon', l.epsilon, ...
          'aggregate', l.aggregate, ...
          'instanceWeights', l.instanceWeights) ;
      
      case 'send'
        res(i).dzdx = res(i+1).dzdx;
        
%         for j = i+1:n
%             l_ = net.layers{j};
%             if (strcmp(l_.type, 'receive') & find(l_.receive == l.send))
%                 res(i).dzdx = res(i).dzdx + res(j+1).dzdx;
%             elseif (strcmp(l_.type, 'concat') & find(l_.concat == l.send))
%                 ch          = 1:size(res(i).dzdx, 3);
%                 ci          = find(l.send == l_.concat);
% 
%                 offset      = 0;
% 
%                 for chi = l_.concat(1:ci-1)
%                     offset      = offset + size(res(chi).x, 3);
%                 end
% 
%                 res(i).dzdx = res(i).dzdx + res(j+1).dzdx(:,:,offset + ch,:);
%             end
%         end
        
        for j = [receive{l.send}, concat{l.send}]
            l_          = net.layers{j};
            
            switch l_.type
                case 'receive'
                    res(i).dzdx = res(i).dzdx + res(j+1).dzdx;
                case 'concat'
                    ch          = 1:size(res(i).dzdx, 3);
                    ci          = find(i == send{l_.concat});
                    
                    offset      = 0;
                    
                    for chi = [send{l_.concat(1:ci-1)}]
                        offset      = offset + size(res(chi).x, 3);
                    end

                    res(i).dzdx = res(i).dzdx + res(j+1).dzdx(:,:,offset + ch,:);
            end
        end
        
      case 'receive'
        res(i).dzdx = res(i+1).dzdx;
          
      case 'concat'
        ch          = size(res(i).x, 3);
        res(i).dzdx = res(i+1).dzdx(:,:,end-ch+1:end,:);

      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;

    end % layers

    switch l.type
      case {'conv', 'convt', 'bnorm'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
    end
    if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
      res(i+1).dzdx = [] ;
      res(i+1).x = [] ;
    end

    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if i > 1 && i == backPropLim && opts.conserveMemory && ~net.layers{i}.precious && isempty(find(cell2mat([receive; concat]) == i + 1, 1))
    res(i).dzdx = [] ;
    res(i).x = [] ;
  end
end
