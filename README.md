# pc_upsampling
GC-PCU: Geometry-Consistent Point Cloud Upsampling
by Dandan Ding, Chi Qiu and Zhigeng Pan

Introduction: 
This work is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators.
The code is tested under TF 1.11.0 and Python 3.5 on Ubuntu 16.04, higher verison should also work.

For compling TF operators, please check tf_xxx_compile.sh under each op subfolder in tf_ops folder. Note that you need to update nvcc, python and tensoflow include library if necessary. For more details, please refer previous work such as PU-Net, MPU and PU-GAN. 

Evaluate the model: We've provided the pre-train model in folder /model, so just run:
cd Code
python main.py --phase test
You will see the output results in the folder /result.

Besides we also provide a python script in folder /supplement which put our model in GAN as generator, if you want to test it, just download the PU-GAN's project from https://github.com/liruihui/PU-GAN and replace original generator.py. 

If you want training the model please contact us for training dataset by email: qiuchi@stu.hznu.edu.cn.
More questions please contact qiuchi@stu.hznu.edu.cn
