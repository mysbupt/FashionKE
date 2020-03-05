# FashionKE
Code of paper: ["Who, Where, and What to Wear? Extracting Fashion Knowledge from Social Media"](https://dl.acm.org/doi/pdf/10.1145/3343031.3350889)

### Requirements
python3\s\s
pytorch 0.4.1\s\s
torchvision\s\s
pyyaml\s\s
json\s\s
numpy\s\s
pickle\s\s
tensorboard\_logger\s\s
scipy\s\s

### Train
You can modify the config file in config.yaml to change the hyper-parameters.

python train.py 

### Log and Results
The default tensorboard log files are saved at ./checkpoint, where you can start a tensorboard to see the training curve. \s\s
The default trained model parameters are saved at ./params. \s\s
The default prediction results are saved at ./result.
