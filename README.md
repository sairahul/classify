# Background

Classify/Xplore is a tool designed for understanding, analyzing and annotating medical images like Xrays ,MRIs, CT scans.

For this demo i already uploaded a dataset of X-Ray images. The dataset is from a kaggle competition. The task is to analyze if the person has pneumothorax or not.

On the Left hand we are rendering the thumbnails. We can click on a image and see its original image on the right hand side pane. What we are rendering here is the original dicom image. The bottom right hand side shows the predictions and the related images. We also see the extracted meta content in the bottom.

For this demo purpose i wrote a small network that computes the similar images using Siamese network. We can plugin any model here.

We can apply different filters while analyzing the image like rotating, inverting, flipping etc.

# ToDo

It is still work in progress. I am working on to bring up the tool without any manual steps.

# References

* [Kaggle Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
* [fast.ai](http://fast.ai)
* [Fastapi](https://github.com/tiangolo/fastapi)
* [Fastapi base project generator](https://github.com/tiangolo/full-stack-fastapi-postgresql)
* [resnet34 model](https://github.com/sairahul/notebooks/blob/master/Siamese_Network_with_ContrastiveLoss.ipynb)


# Demo

http://148.251.87.118:8080

