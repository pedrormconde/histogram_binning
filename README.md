# Histogram Binning for Classification and Object Detection

_Histogram binning_ [1] is a non-parametric method commonly used to calibrate uncertainty estimates of classification models. This method leverages the principle of bin-wise calibration, by substituting the confidence scores of a model by the respective _accuracy per bin_ values, found in a previously defined training set. This calibration method can also be adapted to object detection problem, by considering the concept of _precision per bin_, and shows positive effects in terms of D-ECE [2].

The code for the _classification_ version is present in the file _histogram_binning.py_.  
