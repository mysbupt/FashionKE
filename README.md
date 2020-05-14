# FashionKE

### Introduction
This repo is the code of paper: ["Who, Where, and What to Wear? Extracting Fashion Knowledge from Social Media"](https://dl.acm.org/doi/pdf/10.1145/3343031.3350889), which aims to extract {Cloth, Person, Occasion} -- triplet-formatted fashion knowledge -- from social media.

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

### Dataset
I am very sorry that we cannot release the dataset because the data are collected from Instagram users and we are unable to get all the users' authorization, even though it cost much money to manually check and fix the wrong labels. Hope the model and code of this paper will help or inspire you.

### Train
You can modify the config file in config.yaml to change the hyper-parameters.

python train.py 

### Log and Results
The default tensorboard log files are saved at ./checkpoint, where you can start a tensorboard to see the training curve.   
The default trained model parameters are saved at ./params.   
The default prediction results are saved at ./result.

### Citation
If you use the code of this repo, kindly cite it as:  
```
@inproceedings{ma2019and,
  title={Who, Where, and What to Wear? Extracting Fashion Knowledge from Social Media},
  author={Ma, Yunshan and Yang, Xun and Liao, Lizi and Cao, Yixin and Chua, Tat-Seng},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={257--265},
  year={2019}
}
```

### Acknowledgement
This project is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/mysbupt/FashionKE/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
