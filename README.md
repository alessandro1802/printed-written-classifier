# Printed / Written scanned text document classification

The training took place on *Google Colab* using *RVL-CDIP-small* dataset with 200 examples per class on *invoice* as *printed* and *handwritten* classes.

# Set up
```shell
python3.8 -m venv venv 
source venv/bin/activate
pip install -r requiements.txt
```

# Inference
Change the `image_path` variable in the `inference.py`.  Run with:
```shell
python3 inference.py
```
The assertion at the end is added to verify the results.


# References
* [Original repo](https://github.com/Cliche1998/Binary_Classification_Handwritten_Printed_Images) taken as the base
* [Dataset](https://huggingface.co/datasets/vaclavpechtor/rvl_cdip-small-200/tree/main)


