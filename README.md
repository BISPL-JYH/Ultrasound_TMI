Paper
===============
* Efficient B-mode Ultrasound Image Reconstruction from Sub-sampled RF Data using Deep Learning
  * In revision process (2018): [https://arxiv.org/abs/1712.06096]

Implementation
===============
* MatConvNet (matconvnet-1.0-beta24)
  * Please run the matconvnet-1.0-beta24/matlab/vl_compilenn.m file to compile matconvnet.
  * There is instruction on "http://www.vlfeat.org/matconvnet/mfiles/vl_compilenn/"
  * Please run the installation setup (install.m) and run some training examples.
 
Trained network
===============
* Trained network for 'SC2xRX4 (down-sampling) CNN' is uploaded.

Training/Test data
===============
* Pre-processed training ('DNN4x1_TrVal.mat') and test ('DNN4x1_TestVal.mat') files are placed in 'data' and 'data\cnn_sparse_view_init_multi_normal_dsr2_input64' folders, respectively.
* The dimension of data are as follows
  -- Training_data  =  64x384x1x22731 (channel x scanline x frame x depth)
  -- Test_data      =  64x384x1x2304  (channel x scanline x frame x depth)
                        
To perform a test using proposed algorithm
-> Use 'DNN4x1_TestVal' as input data

-> Run 'MAIN_RECONSTRUCTION.m

-> You can get the reconstructed RF data in the 'data\cnn_sparse_view_init_multi_normal_dsr2_input64' directory.

-> Using standard delay-and-sum (DAS) beam-forming code ('http://www.ultrasoundtoolbox.com/') construct a B-mode image.
