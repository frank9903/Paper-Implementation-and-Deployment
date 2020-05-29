# Paper-Implementation-and-Deployment

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
  * https://github.com/hollance/YOLO-CoreML-MPSNNGraph.git (Comparison Model)
* Getting Started: Git clone this repo and upload [My Mask RCNN.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask%20RCNN/My_Mask_RCNN.ipynb) to __Google Colab__ and run all cells in the notebook
* Deployment Method: __Local Server__
  * Local Inference Model Setup:
  ```
  cd Mask\ RCNN/
  pip install -r 'requirements.txt' 
  python setup.py install
  ```
  * Local Server Setup:
  ```
  cd Local\ Server/
  python manage.py migrate
  cd myproject/
  python manage.py runserver 0.0.0.0:8000
  ```
  Don't forget to change the IP Address defined in the beginning of [FirstViewController.swift](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Deployment/Deployment/View%20Controllers/1st/FirstViewController.swift)
* Presentation:

  <img src="https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask RCNN/demo/IMG_253A54215DE1-1.jpeg" width="480" height="360" />
  <img src="https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask RCNN/demo/ezgif.com-video-to-gif.gif" width="480" height="360" />
  
 ### Image Captioning
 
* Paper References: 
  * [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)
  
* Implementation Reference:
  * https://github.com/Hvass-Labs/TensorFlow-Tutorials.git (TensorFlow Verision)
  * https://github.com/yunjey/pytorch-tutorial.git (Pytorch Version)
  
* Getting Started: Git clone this repo and upload [Image Caption.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Image%20Captioning/Image%20Caption.ipynb) to __Google Colab__ and run all cells in the notebook

  Note that since `onnx_coreml` currently doesn't support RNN conversion, the notebook doesn't have codes to convert Pytorch model to CoreML model.

* Deployment Method: __TensorFlow Model to CoreML Model__, with all the conversions can be found in [Image Caption.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Image%20Captioning/Image%20Caption.ipynb) `Convert Model` section.

* Presentation

  <img src="https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Image%20Captioning/demo/ezgif.com-video-to-gif.gif" width="480" height="360" />
  
  Note that our deployment features auto completion for user input as shown in the last three examples in the gif. This feature could potentially be used to:

  * Asking for more details, i.e. with trees in the background in 3rd example
  * Changing focus of captions, i.e. a bathroom -> a man in 4th example
  * Correct model's mistake, i.e. a bathroom -> kitchen in 5th example
