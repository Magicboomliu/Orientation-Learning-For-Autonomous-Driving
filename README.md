# OrientationLearning
Simple Orientation Learning using ResNets 

- Download the VehicleOrientationDataset  
https://github.com/sekilab/VehicleOrientationDataset  

Note that in this repo, we using consider the category of 'car', which means 3 categories: 
- car_back : 0 
- car_front :1
- car_side :2 



for the training, we use `vehicle-orientation 1~4` for training(almost 20K images), and use `vehicle-orientation-5` for evaluation.


For trainining, we use: 
```
cd scripts 
sh train.sh
```
