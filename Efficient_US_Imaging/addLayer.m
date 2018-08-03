function dst  = addLayer(type, varargin)

ai      = 1;

params  = {};
values  = {};

while ai <= length(varargin)
    params{end + 1}     = varargin{ai}; ai = ai + 1;
    values{end + 1}     = varargin{ai}; ai = ai + 1;
end

switch type
    case 'conv'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'filter';  filter  = values{i};
                case 'output';	output  = values{i};
                case 'stride';  stride  = values{i};
                case 'pad';     pad     = values{i};
                case 'f';       f       = values{i};
            end
        end
        
        opts.method = 'xavier';
        opts.scale  = f;
        
        weights = init_weight(opts, filter, output);
        
        layer_      = struct('type', type, ...
                        'weights', {{weights, zeros(output, 1, 'single')}}, ...
                        'stride', stride, 'pad', pad);
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
        
    case 'convt'
        for i = 1:length(params)
            switch params{i}
                case 'net';         dst         = values{i};
                case 'filter';      filter      = values{i};
                case 'output';      output      = values{i};
                case 'upsample';    upsample    = values{i};
                case 'crop';        crop        = values{i};
                case 'numGroups';	numGroups	= values{i};
                case 'f';           f           = values{i};
            end
        end
        
        opts.method = 'xavier';
        opts.scale  = f;
        
        weights = init_weight(opts, filter, output);
        
        layer_      = struct('type', type, ...
                        'weights', {{weights, zeros(output, 1, 'single')}}, ...
                        'upsample', upsample, 'crop', crop);
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
        
    case 'bnorm'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'opts';    opts    = values{i};
                case 'f';       f       = values{i};
                case 'mu';      mu      = values{i};
                case 'sgm';     sgm     = values{i};
            end
        end
        
        if (opts.batchNormalization)
            assert(isfield(dst.layers{end}, 'weights'));
            ndim = size(dst.layers{end}.weights{1}, 4);
            layer_ = struct('type', type, ...
                'weights', {{f*ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), [mu*ones(ndim, 1, 'single'), sgm*ones(ndim, 1, 'single')]}}, ...
                'learningRate', [1 1 0.05], ...
                'weightDecay', [0 0]) ;
            dst.layers{end}.biases = [] ;
            dst.layers = horzcat(dst.layers(1:end), layer_) ;
        end
        
    case 'relu'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'opts';    opts    = values{i};
            end
        end
        
        layer_      = struct('type', type) ;
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
        
    case 'threshold'
        method  = 'relu';
        th      = 0;
        
        for i = 1:length(params)
            switch params{i}
                case 'method';  method  = values{i};
                case 'th';      th      = values{i};
            end
        end
        
        dst     = struct('type', type, ...
                        'method', method, ...
                        'th', th) ;
                        
    case 'pool'
        for i = 1:length(params)
            switch params{i}
                case 'method';  method  = values{i};
                case 'pool';	pool    = values{i};
                case 'stride';  stride  = values{i};
                case 'pad';     pad     = values{i};
                case 'net';     dst     = values{i};
            end
        end
        
        layer_      = struct('type', type, ...
                        'method', method, ...
                        'pool', pool, ...
                        'stride', stride, ...
                        'pad', pad) ;
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
                    
    case 'unpool'
        for i = 1:length(params)
            switch params{i}
                case 'method';  method  = values{i};
                case 'pool';	pool    = values{i};
                case 'stride';  stride  = values{i};
                case 'pad';     pad     = values{i};
                case 'ipool';   ipool   = values{i};
                case 'net';     dst     = values{i};
            end
        end
                     
        if (ipool > 0 && ~strcmp(dst.layers{ipool}.type, 'pool'))
            error(['DO NOT CONNECT WITH No.' num2str(ipool) ' LAYER.']);
        end
        
        layer_      = struct('type', type, ...
                        'method', method, ...
                        'pool', pool, ...
                        'stride', stride, ...
                        'pad', pad, ...
                        'ipool', ipool) ;
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
                    
    case 'send'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'send';    send	= values{i};
            end
        end
        
        l   = length(dst.layers) + 1;
        
        layer_      = struct('type', type, ...,
                            'send', send);
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
                        
    case 'receive'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'receive';	receive	= values{i};
            end
        end
        
        l   = length(dst.layers) + 1;
                
%         for j = 1:length(receive)
%             if (~isfield(dst.layers{receive(j)}, 'send') && isempty(find(dst.layers{receive(j)}.send(:) == l, 1)))
%                 error(['DO NOT CONNECT WITH  BETWEEN No.' num2str(concat(j)) '& No. ' num2str(l) ' LAYERS.']);
%             end
%         end
        
        layer_      = struct('type', type, ...,
                            'receive', receive);
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
                        
    case 'concat'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'concat';	concat	= values{i};
            end
        end
        
        l   = length(dst.layers) + 1;
        
%         for j = 1:length(concat)
%             if (~isfield(dst.layers{concat(j)}, 'send') || isempty(find(dst.layers{concat(j)}.send(:) == l, 1)))
%                 error(['DO NOT CONNECT WITH  BETWEEN No.' num2str(concat(j)) '& No. ' num2str(l) ' LAYERS.']);
%             end
%         end
            
        layer_      = struct('type', type, ...,
                            'concat', concat);
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
        
    case {'wavedec', 'waverec'}
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
                case 'lv';      lv      = values{i};
                case 'pflt'; 	pflt 	= values{i};
                case 'dflt'; 	dflt 	= values{i};
            end
        end
        
        layer_      = struct('type', type, ...
                        'lv', lv, ...
                        'pflt', pflt, ...
                        'dflt', dflt) ;
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
                        
    case 'euclideanloss'
        for i = 1:length(params)
            switch params{i}
                case 'net';     dst     = values{i};
            end
        end
        
        layer_     	= struct('type', type) ;
        dst.layers	= horzcat(dst.layers(1:end), layer_) ;
        
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, filter, output, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

if nargin < 4
    type = 'single';
end

h   = filter(1);
w   = filter(2);
in	= filter(3);

switch opts.method
  case 'gaussian'
    sc = opts.scale ;
    weights = randn(h, w, in, output, type)*sc;
  case 'xavier'
    sc = opts.scale*sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, output, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = opts.scale*sqrt(2/(h*w*output)) ;
    weights = randn(h, w, in, output, type)*sc ;
end