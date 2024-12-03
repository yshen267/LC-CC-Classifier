# LC-CC-Classifier
ML classifier project which uses histopathological images of benign and cancerous lung and colon cancer.

Directly download the processed dataset to skip converting images to histograms.
https://drive.google.com/file/d/1tcWHbaxAUF7X56ijrOfWwBq48HA0uSqS/view?usp=sharing

All files should be in the same current working directory

Data Processing.ipynb:
Jupyter notebook showing example of image processing pipepline and responsible for running the processing steps in parrell in conjunction with worker_script.

For multiprocessing, the number of workers is currently set to os.cpu_count(). Reduce this number if it is too taxing on your machine.

LCC_Classifier.ipynb:
Jupyter notebook containing processed image data exploration and model training.

tuner_dir.zip:
This contains the CNN trained as a comparison model. This should be unzipped and put into the current working directory as well.
