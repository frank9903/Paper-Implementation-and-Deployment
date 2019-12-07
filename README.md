# Paper-Implementation-and-Deployment
Build, train and deploy papers
## Introduction
This repo is not meant to exactly reproduce the paper results but instead, what I want to achieve is to understand and visualize paper models without tedious environment setups. To do so, I'm going to build, train and validate the models all in __Google Colab__. On top of that, this repo will also focus on various ways of deploying the models (for iOS), including but not limited to:
1. Use local server and HTTP request
2. Convert FastAI/Pytorch model to CoreML model
3. Convert Keras/TensorFlow model to CoreML model 

***Colab tips: In order to keep Colab from disconnecting, add the following to your browser console.***
```
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
```
More details can be found [here](https://medium.com/@shivamrawat_756/how-to-prevent-google-colab-from-disconnecting-717b88a128c0).
## Table of Contents
### Mask R-CNN
* Paper References: 
  * Original Paper: [Mask R-CNN](https://arxiv.org/pdf/1703.06870v3.pdf)
  * Related Paper: [R-CNN](https://arxiv.org/pdf/1311.2524.pdf), [Fast R-CNN](https://arxiv.org/pdf/1504.08083v2.pdf), [Faster R-CNN](https://arxiv.org/pdf/1506.01497v3.pdf)
* Implementation Reference: 
  * https://github.com/matterport/Mask_RCNN.git
  * https://github.com/sunshineatnoon/IOS_UPLOAD_TO_DJANGO_DEMO.git
* Getting Started: Git clone this repo and open [My Mask RCNN.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask%20RCNN/requirements.txt) using __Google Colab__ to run all cells in the notebook
* Deployment Method: __Local Server__
  * Local Server Setup:
  ```
  cd Local\ Server/
  python manage.py migrate
  cd myproject/
  python manage.py runserver 0.0.0.0:8000
  ```
  Don't forget to change the IP Address defined in the beginning of [FirstViewController.swift](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Deployment/Deployment/FirstViewController.swift)
* Presentation: ___TODO___
