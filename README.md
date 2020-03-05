# FashionKE
Code of paper: ["Who, Where, and What to Wear? Extracting Fashion Knowledge from Social Media"](https://dl.acm.org/doi/pdf/10.1145/3343031.3350889)

### Requirements
python3  
pytorch 0.4.1  
torchvision  
pyyaml  
json  
numpy  
pickle  
tensorboard\_logger  
scipy  

### Train
You can modify the config file in config.yaml to change the hyper-parameters.

python train.py 

### Log and Results
The default tensorboard log files are saved at ./checkpoint, where you can start a tensorboard to see the training curve.   
The default trained model parameters are saved at ./params.   
The default prediction results are saved at ./result.
