# Histogram Binning for Classification and Object Detection

_Histogram binning_ [1] is a non-parametric method commonly used to calibrate uncertainty estimates of classification models. This method leverages the principle of bin-wise calibration, by substituting the confidence scores of a model by the respective _accuracy per bin_ values, found in a previously defined training set. This calibration method can also be adapted to object detection problem, by considering the concept of _precision per bin_, and shows positive effects in terms of D-ECE [2].

The code for the _classification_ version is present in the file _histogram_binning.py_.  To train (_histogram_binning_train_ function) the method, the inputs (1st and 2nd argument) are _numpy arrays_ (or _torch tensors_) of dimension $(N,k)$ and $(N)$, repectively, where $N$ is he number of samples of that training set and $k$ the number of classes. The 1st argument is the output of the classification model (probability vectors, _i.e. after the _softmax_ function) while the 2nd argument is a N-dimensional vector with the respective ground-truth classes. To use the method for inference in a test set (_histogram_binning_predict_ function) the input is a $(M,k)$  _numpy array_ (or _torch tensor_) of probabability vectors outputs, where $M$ is the size of that test set.

The code for the _object detection_ version is in the _histogram_binning_od.py_ file. To train the method (_HB_train_ function) the 1st an 2nd arguments are, respectively, the directory of a folder with all the predicted bounding-box text files, and the directory of a folder with all the ground-truth bounding-box text files, with respect to the chosen training set. For inference (_HB_predict function), the 1st argument is the directory of a folder with all the predicted bounding-box text files, for the chosen test set. All text files must be in the common YOLOv5 format.

## References

[1] Zadrozny, B., Elkan, C.: Obtaining calibrated probability estimates from decision trees and Naive Bayesian classifiers. In: International Conference on Machine Learning (ICML). vol. 1, pp. 609–616. Citeseer (2001)

[2] Kuppers, F., Kronenberger, J., Shantia, A., Haselhoff, A.: Multivariate confidence calibration for object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 326–327 (2020)
