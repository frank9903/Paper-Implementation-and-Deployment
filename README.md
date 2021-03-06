# Paper-Implementation-and-Deployment

## Introduction
This repo is not meant to exactly reproduce the paper results but instead, what I want to achieve is to understand and visualize paper models without tedious environment setups. To do so, I'm going to build, train and validate the models all in __Google Colab__. On top of that, this repo will also focus on various ways of deploying the models (for iOS), including the following:
1. Use local server and HTTP request
2. Convert Pytorch model to CoreML model
3. Convert TensorFlow model to CoreML model 
4. Use Turi Create to train and deploy model

***Colab tips: In order to keep Colab from disconnecting, add the following to your browser console.***
```
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```
Please keep track of [this thread](https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting) for more detailed infomation and updates.
## Table of Contents
### Object Detection [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iBUNsBCqqKZ-OmlQdLXx-yrwlYx4sFmT?usp=sharing)
* Paper References: 
  * Original Paper: [Mask R-CNN](https://arxiv.org/pdf/1703.06870v3.pdf)
  * Related Paper: [R-CNN](https://arxiv.org/pdf/1311.2524.pdf), [Fast R-CNN](https://arxiv.org/pdf/1504.08083v2.pdf), [Faster R-CNN](https://arxiv.org/pdf/1506.01497v3.pdf)
* Implementation Reference: 
  * https://github.com/matterport/Mask_RCNN.git
  * https://github.com/sunshineatnoon/IOS_UPLOAD_TO_DJANGO_DEMO.git
  * https://github.com/hollance/YOLO-CoreML-MPSNNGraph.git (Comparison Model)
* Getting Started: Click on `Open In Colab` button above (or upload [My Mask RCNN.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask%20RCNN/My_Mask_RCNN.ipynb) to __Google Colab__) and run all cells in the notebook
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
  Don't forget to change the IP Address defined in the beginning of [Mask RCNN Controller.swift](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Deployment/Deployment/View%20Controllers/Mask%20RCNN/Mask%20RCNN%20Controller.swift)
* Presentation:

  <img src="https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Mask RCNN/demo/ezgif.com-video-to-gif.gif" width="480" height="360" />
  
 ### Image Captioning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13jxlrfvYapTNGItR38ilnDqhzknZvWug?usp=sharing)
 
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

### Style Transfer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xjCvNnSH307XB3PZcT1COBnsIEPO63Yx?usp=sharing)

* Paper References: 
  * [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)
  * [A neural algorithm of artistic style](https://arxiv.org/pdf/1508.06576.pdf)
  
* Implementation Reference:
  * https://github.com/eriklindernoren/Fast-Neural-Style-Transfer (Pytorch Verision)
  * https://apple.github.io/turicreate/docs/userguide/style_transfer/ (Turi Create Version)
 
* Deployment Method: __Pytorch Model to CoreML Model__ and __Turi Create Model__, with all the conversions can be found in [Style Transfer.ipynb](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Style%20Transfer/Style%20Transfer.ipynb)

* Presentation

  <img src="https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Style%20Transfer/demo/ezgif.com-video-to-gif.gif" width="480" height="360" />
  
  Note that a scene detector is added to our deployment because some styles are not appliable to all images (e.g. `Starry Night` style is not working well on images with people). If you want to add new style transfer model to the App, please remember to add constraint to `UNEXPECTED_CONTENT` in [Style Transfer Controller](https://github.com/shuheng-cao/Paper-Implementation-and-Deployment/blob/master/Deployment/Deployment/View%20Controllers/Style%20Transfer/Style%20Transfer%20Controller.swift).
