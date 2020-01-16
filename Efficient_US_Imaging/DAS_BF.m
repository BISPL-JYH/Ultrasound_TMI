clc
clear all
close all

dataFile = ['data\cnn_sparse_view_init_multi_normal_dsr2_input64\DNN4x1_TestVal.mat'];
load(dataFile);

rec = images.data; % change this for images.label for label and for reconstructed image use rec

[nNumCh,ScanlineNum,Numframes,AlignedSampleNum]=size(rec);
Reconstruction = permute(2048*rec,[1 4 2 3]);

%%
Rx_F_num = 1;
Offset = 50;
fs = 40e6;
c = 1540;

% N_ele = double(System.Transducer.elementCnt);
N_ele = 192;
% pitch = double(System.Transducer.elementPitchCm) * 1e-2;     % cm => m
pitch = 0.0200 * 1e-2;     % cm => m

AlignedSample = double(AlignedSampleNum);
SampleNum = 2469;
DepSample = double(SampleNum);

nNumCh = 64;
N_ch = nNumCh;
nHalfNumCh = nNumCh/2;
sam_st_2nd = AlignedSample*nHalfNumCh;

data_total = AlignedSample;
data_total1 = DepSample;
pixel_d = c/fs/2;

%%
scan_view_size = pitch*N_ele; % Lateral View Size

N_sc = double(ScanlineNum);
% N_sc = ((scan_view_size/pitch)) ; % Scanline number
sc_d = scan_view_size/(N_sc); % Scanline distance
st_sc_x = - scan_view_size/2+sc_d/2 ; % Start Scanline position

st_sam = round(0.001/pixel_d);
ed_sam = data_total1;
%% DC cancle filter
f = [0 0.1 0.1 1]; m = [0 0 1 1];
DC_cancle = fir2(64,f,m);
%% rf channel data interpolation
interp_rate = 1;%8; % channel data interpolation rate
%% reordering information
rx_HalfCh = N_ch*0.5;
rx_ch_mtx = [-rx_HalfCh:rx_HalfCh-1];

RxMux = zeros(N_sc, N_ch);
SCvsEle = sc_d/pitch;

for sc = 1:N_sc
    idx = floor((sc-1)*SCvsEle) + 1;

    rx_idx_tmp = idx + rx_ch_mtx;
    rx_idx = rx_idx_tmp((rx_idx_tmp > 0) & (rx_idx_tmp <= N_ele));
    RxMux(sc,:) = rx_idx_tmp;
end
%%
half_ele = -(N_ele-1)*pitch/2; % half aperture size
half_rx_ch = N_ch*pitch/2; % half rx channel size

sc_weight = zeros(data_total,1);
sc_weight(st_sam:ed_sam) = 1;

n_sample = [0:data_total-1]' + Offset;
d_sample = n_sample * pixel_d;
d_mtx = d_sample * ones(1,N_ele); % depth information matrix

ele_pos = [half_ele:pitch:-half_ele]; % element position matrix

% rx data read pointer offset matrix
rp_mtx = ones(data_total,1) * [0:data_total*interp_rate:data_total*interp_rate*(N_ele-1)];

LPBF_output  = zeros(data_total, N_sc);

j = sqrt(-1);
%%

%%    
for sc = 1:N_sc
    disp(['SC : ',num2str(sc)])
    cmd = sprintf('ChData = Reconstruction(:,:,sc)'';');
    eval(cmd);

    RF_Ch_data = zeros(data_total, N_ele);

    MuxTbl = RxMux(sc,:);
    idx = find((MuxTbl > 0) & (MuxTbl <= N_ele));
    idx1 = MuxTbl(idx);

    RF_Ch_data(:,idx1) = ChData(:,idx);

    %    figure(1)
    %    imagesc(abs(ChData));
    %    figure(2)
    %    imagesc(abs(RF_Ch_data));
    %%    
    inter_p_B_tmp = zeros(data_total*interp_rate,N_ele);

    for ch = 1:N_ele
        tmp = RF_Ch_data(:,ch);
        fil_tmp = conv(tmp, DC_cancle, 'same');
        inter_p_B_tmp(:,ch) = interp(fil_tmp,interp_rate);
    end
    %     inter_p_B_tmp = hilbert(inter_p_B_tmp); % hilbert transform

    Rx_apod_index = zeros(data_total, N_ele); % apodization matrix initialize

    sc_pos = st_sc_x + (sc-1)*sc_d; % current scanline position

    ch_pos = abs(sc_pos - ele_pos); % current channel position
    ch_pos_mtx = ones(data_total,1) * ch_pos;

    %%%%%%%%%%% Apodization matrix generation %%%%%%%%%%%%%%%%
    aper_size = d_mtx/Rx_F_num; % dynamic aperture size
    h_aper_size = aper_size * 0.5; % half size

    idx = find(h_aper_size >= half_rx_ch); % full aperture region index
    idx1 = find(h_aper_size < half_rx_ch); % small aperture region index

    apo_tmp = ch_pos_mtx./half_rx_ch;
    apo_tmp1 = ch_pos_mtx./h_aper_size;

    Rx_apod_index(idx) = apo_tmp(idx);
    Rx_apod_index(idx1) = apo_tmp1(idx1);

    idx4 = find(Rx_apod_index >= 1);
    Rx_apod_index(idx4) = 1;

    Rx_apod_index_r = Rx_apod_index;
    idx5 = find(Rx_apod_index_r < 1);
    Rx_apod_index_r(idx5) = 0;

    rect_apo = 0.5+0.5.*cos(Rx_apod_index_r.*pi);    % rectangle window

    %%%%%%%%%%%% focused channel data %%%%%%%%%%%%%%%%%%%

    tx_t = d_mtx./c;
    rx_t = sqrt(ch_pos_mtx.^2 + d_mtx.^2)./c;

    read_pointer_rx = round((tx_t + rx_t)*fs*interp_rate);

    % 예외 처리
    idx = find(read_pointer_rx > data_total*interp_rate);
    read_pointer_rx(idx) = data_total*interp_rate;

    idx = find(read_pointer_rx < 1);
    read_pointer_rx(idx) = 1;

    focused_ch_data = conj(inter_p_B_tmp(read_pointer_rx+rp_mtx)');
    LPBF_output(:, sc) = sum(focused_ch_data);
end    

%     filename = sprintf('%s/LPBF_output.mat', pname);
%     save(filename,'LPBF_output');

%%
LPBF_output = hilbert(LPBF_output); % hilbert transform

RF_env = abs(LPBF_output);

data_max = max(max(RF_env));
% data_max = 751.3949;

log_data = RF_env./data_max;

dB = 60;
min_dB = 10^(-dB/20);

for i=1:N_sc
    for j=1:data_total
        if(log_data(j,i) < min_dB)
            log_data(j,i) = 0;
        else
            log_data(j,i) = 255*((20/dB)*log10(log_data(j,i))+1);
        end
    end
end

%%
img_width = scan_view_size;
img_dep = data_total*pixel_d;

img_x = 500;
img_z = 650;

B_img = zeros(img_z, img_x);

dx = img_width/(img_x-1);
dz = img_dep/img_z;

for i=1:img_x
    ix = (i-1)*dx;
    for j=1:img_z
        iz = (j-0.5)*dz;

        z = iz/pixel_d;
        x = ix/sc_d + 1;

        z_L = floor(z);
        z_H = z_L+1;
        x_L = floor(x);
        x_H = x_L+1;

        z_err = z-z_L;
        x_err = x-x_L;

        if((z_L>0) && (z_H <= data_total) && (x_L > 0) &&(x_H <= N_sc))
            Zon = log_data(z_L,x_L);
            Zon1 = log_data(z_H,x_L);
            Zin = log_data(z_L,x_H);
            Zin1 = log_data(z_H,x_H);

            Zri = Zin*(1-z_err) + Zin1*z_err;
            Zro = Zon*(1-z_err) + Zon1*z_err;
            Z = Zro*(1-x_err) + Zri*x_err;

            B_img(j,i) = Z;
        end
    end
end

B_img = B_img(1:500,:);

filename = ['RF_sum_DSC_dB',num2str(dB),'.bmp'];
imwrite(uint8(B_img),filename,'bmp');
