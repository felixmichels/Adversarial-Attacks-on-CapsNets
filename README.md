# Adversarial-Attacks-on-CapsNets

This repository contains code for my bachelor thesis "Adversarial Attacks on Capsule Networks" as well as for the paper that was created in the process (https://arxiv.org/abs/1906.03612).

All code for defining and training the models, generating adversarial examples and extracting data is in the 'code' folder.

### Accuracies and model names

| Model/Dataset | mnist               | fashion               | svhn                  | cifar10          |
|---------------|---------------------|-----------------------|-----------------------|------------------|
| ConvNet       | conv_baseline<br>99.4% | conv_baseline<br>92.9%   | conv_baseline<br>92.6%   | conv_good<br>88.2%  |
| CapsNet       | capsnet_small<br>99.4% | capsnet_variant<br>92.7% | capsnet_variant<br>92.4% | dcnet<br>88.2%      |


### Example images:  
Left original, middle adversarial example, right magnified perturbation  
Top Capsule Network, bottom Convolutional Network  

#### Carlini-Wagner:

![alt text](resources/img/zoomed/cw_orig_automobile.png)
![alt text](resources/img/zoomed/cw_caps_frog.png)
![alt text](resources/img/zoomed/cw_caps_pert.png)  

![alt text](resources/img/zoomed/cw_orig_automobile.png)
![alt text](resources/img/zoomed/cw_conv_frog.png)
![alt text](resources/img/zoomed/cw_conv_pert.png)  
  
  
#### Boundary-Attack:  

![alt text](resources/img/zoomed/boundary_orig_horse.png)
![alt text](resources/img/zoomed/boundary_caps_48.png)
![alt text](resources/img/zoomed/boundary_caps_pert.png)  

![alt text](resources/img/zoomed/boundary_orig_horse.png)
![alt text](resources/img/zoomed/boundary_conv_48.png)
![alt text](resources/img/zoomed/boundary_conv_pert.png)
  
  
#### Deepfool:  

![alt text](resources/img/zoomed/deepfool_orig_dog.png)
![alt text](resources/img/zoomed/deepfool_caps_95.png)
![alt text](resources/img/zoomed/deepfool_caps_pert.png)  

![alt text](resources/img/zoomed/deepfool_orig_dog.png)
![alt text](resources/img/zoomed/deepfool_conv_95.png)
![alt text](resources/img/zoomed/deepfool_conv_pert.png)  
  
  
#### Universal Attack:  

![alt text](resources/img/zoomed/universal_orig_bird.png)
![alt text](resources/img/zoomed/universal_caps_adv.png)
![alt text](resources/img/zoomed/universal_caps_pert.png)  

![alt text](resources/img/zoomed/universal_orig_bird.png)
![alt text](resources/img/zoomed/universal_conv_adv.png)
![alt text](resources/img/zoomed/universal_conv_pert.png)
