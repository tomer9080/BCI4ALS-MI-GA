# BCI4ALS-Motor Imagery Using Genetic Algorithm
Hi everyone, this repository contains the code for BCI4ALS Motor imagery project.

This repo is using Genetic Algorithm (GA) to try and optimize the best features for the classifiers,
in a goal to find the features that can produce better results for the long-term.

 [BCI4ALS MI GA](#BCI4ALS-MI-GA)
  * [Previous Work](#Previous-Work)
  * [Data Acquisition](#Data-Acquisition)
  * [Feature Selection](#Feature-Selection)
  * [Classifier Ensemble](#Classifier-Ensemble)
  * [Optimization](#Optimization)
  * [Results](#Results)
  * [Usage](#Usage)
  * [Further Work](#Further-Work)
  * [Project Book](#Project-Book)


## Previous Work
The matlab base code in thie repo was taken from this repo: (https://github.com/harelasaf/BCI4ALS-MI).
If you want to learn more about the course:  https://www.brainstormil.com/bci-4-als.

Another recommended python package (was not used in this project) is bci4als:
https://pypi.org/project/bci4als/

## Data Acquisition
 The data used in this project was recorded on us over the span of 6 weeks, with 3 sessions each week. Each session is comprised of 60 trials in which the subject will be shown randomly one of
3 option - right, left or idle.

## Data Processing
The acquired data was later Pre-processed and used to extract feature that would help classify the results to get maximum prediction rates success, as shown in the figure below. The classifiers used at the beginning were basic classifiers such as LDA or SVM.\
Further explanation about each stage in the figure can be seen in the project book, the link to it is a the end of this README file.


 <p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/Block_diagram.png" />
</p>

##  Feature Selection
We used the Genetic Algorithm to see which features would yield the best predictions and kept histograms of the selected features.\
After running the algorithm for a large number of times we could know which of the features popped up the most and reduce the feature space.
<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/top_ten_features/NB_top_ten_features.png" />
</p>


## Classifier Ensemble
In order to get the best result we've tried to combine the different basic classifiers into a large meta-classifier in two different ways:
* Expert Advice
* Stacking

Further explanation about this methods can be found here:
- Expert Advice: https://theoryofcomputing.org/articles/v008a006/
- Stacking: https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231

## Optimization
We've used 'Optuna' to tweak the Genetic Algorithm hyper-parameters in order to get the best prediciton rates.

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/important%20params%20full.png" />
</p>

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/param_contour_cross_ind_prob_muta_ind_prob.png" />
</p>


## Result
Comparing the different classifiers and Ensembles can be shown in the figure below:
<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/Results_thresh_0(1).jpg" />
</p>

Once adding the threshold limitation on the features we get a slight increase in the results:

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/Results_thresh_50(1).jpeg" />
</p>

However, a threshold too high will show a start of a decline in the results:

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/Results_thresh_80(1).jpeg" />
</p>

It is noticeable that the KNN classifiers show poor results, and the ensembles results are not changed much if they are removed:

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/Results_thresh_50_no_knn(2).jpeg" />
</p>


The best classifiers over the different thresholds:

<p align="center">
  <img src="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/figures/MVGA_results_on_all(1).jpeg" />
</p>



## Usage
Before running, we recommend to use the file `projenv.yml` in the repo, to setup a sutiable conda enviornment.
After activating the enviornment you can run the below examples. 

To see the full functionality of our script you can use:
```bash
python MI5_ModelTraining.py -h 
```

### Classification Usage Examples:
To run classification on a specific recording, run the command:
```bash
python MI5_ModelTraining.py -f \<recording_path\> -ga \<True/False\> -th \<Threshold on features - must come with ga True\>
```

To run classification on the whole recordings:
```bash
python MI5_ModelTraining.py -pa paths/paths_linux.txt -ga \<True/False\> -th \<Threshold on features - must come with ga True\> -n \<dir_name will be saved in class_results\>
```

And many more options are available.
The above were mainly used by us.

To run the matlab code, and to record data, you can read the documentation inside the matlab code in the Matlab sub-directory.

### Histograms
To get histograms, run:
```bash
python OurGraphs.py
```
The graphs will be saved in the ga_hists directory.

Please feel free to contact us if you've encountered any issues.
## Further Work

The work we presented here achieved good results, but definitely there are aspects to improve and examine such as:
- Finding more helpful features
- Building an ensemble classifier using more base classifiers and assembled by GA.
- Build an online classifier using the GA selected features.
## Project Book
All of our work can be found in detail in the Project Book. <a href="https://github.com/tomer9080/BCI4ALS-MI-GA/blob/master/EEG_BCI4ALS___Project_Summary.pdf">


Hope this was helpful and please let us know if you have any comments on this work:

https://github.com/tomer9080

https://github.com/roilatzres