# Ultrasound_TMI
Ultrasound Image reconstruction

training_data  : 'DNN4x1_TrVal.mat' in 'data' folder
test_data :  In 'cnn_sparse_view_init_multi_normal_dsr2_input64' in 'data' folder.
             'DNN4x1_TestVal.mat'
       
the dimension of data : training_data =  64x384x1x22731  (channelx scanline x frame x depth)
                        test_data =   64x384x1x2304 (channel x scanline x frame x depth)
                        
Test the Algorithm
-> Open 'MAIN_RECONSTRUCTION.m
-> Use 'DNN4x1_TestValat' as input data
-> You can get the reconstructed RF data
-> Beamforming
-> You can get the image
