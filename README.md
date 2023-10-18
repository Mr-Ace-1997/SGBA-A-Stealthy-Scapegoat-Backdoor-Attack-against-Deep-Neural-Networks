# SGBA-A-Stealthy-Scapegoat-Backdoor-Attack-against-Deep-Neural-Networks
This code is the attack scheme in the paper "SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks". We supply this version as a reference of our attack scheme. You can find the paper at <https://doi.org/10.1016/j.cose.2023.103523>.

Among the files, we supply the model structures and data processes of MNIST/GTSRB/CIFAR10/ImageNet(ISLVRC2012). You can download the corresponding datasets by yourself and run our codes on them directly.

You can test the attack scheme as following steps:

1. Generate a clean model to validate the baseline classification accuracy:
   > python clean_train.py --task mnist --to_file

2. Poison the model and generate the special trigger:
   > python train.py --task mnist --to_file --weight_limit

   Note that before you set the argument `weight_limit`, you'd better run a set of clean models and evaluate the applicable limitation settings according to the paper first. Of course, we supply a setting in the code for reference. But to avoid differences caused by the running environment, etc., it's best to replace it with your own estimate.

Feel free to contact me (guapi7878@gmail.com) if you have any questions.
