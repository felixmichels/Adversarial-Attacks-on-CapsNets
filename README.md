# Adversarial-Attacks-on-CapsNets

### Results  
| Avg. Pert Norm | CapsNet | ConvNet |
|----------------|---------|---------|
| Carlini-Wagner | 0.371   | 0.556   | 
| Boundary       | 0.989   | 1.757   |
| Deepfool       | 0.190   | 0.261   |

| Fooling Rate   | CapsNet | ConvNet |
|----------------|---------|---------|
| Carlini-Wagner | 94.8%   | 95.6%   |
| Boundary       | 100%    | 100%    |
| Deepfool       | 100%    | 94.5%   |

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
