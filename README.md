# OrientationLearning
Simple Orientation Learning using ResNets 

- Download the VehicleOrientationDataset  
https://github.com/sekilab/VehicleOrientationDataset  

Note that in this repo, we using consider the category of 'car', which means 3 categories: 
- car_back : 0 
- car_side: 1
- car_front :2




for the training, we use `vehicle-orientation 1~4` for training(almost 20K images), and use `vehicle-orientation-5` for evaluation.

For trainining, we use: 
```
cd scripts 
sh train.sh

```

For inference on new images, we provided the inference code with pre-trained model:

- Download the pretrained model
```
sh download_model.sh
```
The Downloaded model with be in `pretrained_models/model_best.pth`  

Then run the inference code: 
```

python inference.py --pretrained_model_path "pretrained_models/model_best.pth" --nums_layers 34 \
> --image_path "<YOU IMAGE PATH>" --annotation_path "<ANNOTATION JSON FILE>" --saved_folder "<SAVED FOLDER >"

```

