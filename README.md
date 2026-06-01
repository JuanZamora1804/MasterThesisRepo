# MasterThesisRepo

In this repository you can find the main and more relevant, but not all, of the scripts I used in my master thesis.
Feel free to contact me for more information and more scripts.

## 0. Extract image names (0_extract_names.py)
Images should be saved in a folder. Script will extract the images' names and save a .csv file.

## 1. Bridge classification with OpenCLIP (1_openclip_bridge.py)
Images are evaluated in the models that the user defines within the code.
It exports a .csv file with the evaluation metrics and the classification of the images according to the categories defined in the thesis report.

## 2. Image filtering (2_filtering.py)
The classified images are filter according to desired categories (proximity to the deck or image content) and then the evaluation of the overlap for this subset of the images is performed.
As a result, it gives a smaller set of images that can be used in VGGT model.
(overlap_lightglue_analysis.py and overlap_lightglue_utils.py are required)

## 3. Reconstruction and Evaluation.
Contact Mr. Morris Florek, M.Sc. for the modification of VGGT's pipeline addapted to bridges. 
