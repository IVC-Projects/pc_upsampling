# **Point Cloud Upsampling via Perturbation Learning**

by Dandan Ding, Chi Qiu, Fuchang Liu and Zhigeng Pan

Our article has been published in <https://ieeexplore.ieee.org/document/9493165>, more supplementary information can refer to [GC-PCU supplementary information](https://drive.google.com/file/d/18hK_mcmRafjzmJ3w1ZM0LEkofP5XwP9X/view?usp=sharing).

## Introduction: 
This work is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators.
The code is tested under TF 1.11.0 and Python 3.5 on Ubuntu 16.04, higher verison should also work.

For compling TF operators, please check tf_xxx_compile.sh under each op subfolder in tf_ops folder. Note that you need to update nvcc, python and tensoflow include library if necessary. For more details, please refer previous work such as PU-Net, MPU and PU-GAN. 

Evaluate the model: We've provided the pre-train model in folder ./model, so just run:

```
cd Code
python main.py --phase test
```

You will see the output results in the folder ./result.

Besides we also provide a python script in folder ./supplement which put our model in GAN as generator, if you want to test it, just download the PU-GAN's project from https://github.com/liruihui/PU-GAN and replace original generator.py. 

If you want to train the model please download the training dataset from [GoogleDrive](https://drive.google.com/file/d/17aZ9pRi2eqgCIfj-JWA8RPzK2trHzrAd/view?usp=sharing).


More questions please contact <785120046@qq.com> or <qiuchi@stu.hznu.edu.cn>.
