# Ultrasound_TMI
Ultrasound Image reconstruction

Data
===============
* Training_data  :   'DNN4x1_TrVal.mat' 
  * In 'data' folder
* Test_data :  'DNN4x1_TestVal.mat'
  * In 'cnn_sparse_view_init_multi_normal_dsr2_input64' in 'data' folder.
  
       
Dimension of data : 

  * Training_data   =  64x384x1x22731  (channelx scanline x frame x depth)
  
  * Test_data   =   64x384x1x2304 (channel x scanline x frame x depth)
                        
Implementation
===============
-> Open 'MAIN_RECONSTRUCTION.m

-> Use 'DNN4x1_TestValat' as input data

-> You can get the reconstructed RF data

-> Beamforming

-> You can get the image

Network
===============
* Trained Network is uploaded in 'cnn_sparse_view_init_multi_normal_dsr2_input64' in 'data' folder.
 * Network model for 190, 200, 210, 220, 230, 240 epochs
