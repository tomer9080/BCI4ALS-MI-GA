# BCI4ALS-Motor Imagery Using Genetic Algorithm
Hi everyone, this repository contains the code for BCI4ALS Moto imagery proejct.

This repo is using Genetic Algorithm (GA) to try and optimize the best features for the classifiers,
in a goal to find the features that can produce better results for the long-term.

 [BCI4ALS MI GA](#BCI4ALS-MI-GA)
  * [Previous Work](#Previous-Work)
  * [Data Processing](#Data-Processing)
  * [Feature Extraction](#Feature-Extraction)
  * [Feature Selection](#Feature-Selection)
  * [Classifier Selection](#Classifier-Selection)
  * [Results](#Results)
  * [Usage](#Usage)
  * [Files in the Repository](#Files-in-the-Repository)
  * [Further Work](#Further-Work)


## Previous Work
The matlab base code in thie repo was taken from this repo: (https://github.com/harelasaf/BCI4ALS-MI).
If you want to learn more about the course:  https://www.brainstormil.com/bci-4-als.

Another recommended python package (was not used in this project) is bci4als:
https://pypi.org/project/bci4als/

## Data Processing
* We’ve collected data by recording ourselves for a span of 6 weeks.
* After preprocessing the data we extracted different features.
* After analyzing the data we created a feature space of all the used features.

## Architecture
We used PyTorch HuggingFace's RWKV model. The original model was compromised of embedding layer, which we decided to remove, 
since we wanted the NN to handle numerical data. The RWKV model is implemented as in the pubilshed article: [RWKV](https://arxiv.org/pdf/2305.13048.pdf).


The model structure, including our added layers:
<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/rwkv_arch.png" width="450"/>
</p>


## Hyperparameters
* `batch_size` = int, size of batch
* `epochs` = int, number of epochs to run the training
* `window_size` = int, the length of the sequence
* `hidden_states` = int, the number of hidden states in eahc RWKV block
* `hidden_layers` = int, the number of RWKV blocks in the NN.
* `dropout` = float, the dropout probability of the dropout layers in the model (0.0 - 1.0)
* `lr` = float, starting learning rate 
* `factor` = float, multiplicative factor of learning rate decay (0.0 - 1.0)
* `patience` = int, how many epochs we'll wait before decaying lr after no improvement
* `optimizer` = pytorch optimizer, In what method we'll try to optimize our criterion.
* `scheduler` = pytorch scheduler, In what granularity/method we are reducing our lr.


## Result

We trained the model with the hyperparameters:

|Param|Value|
|-------|-------------|
|`window_size` | 40 |
|`hidden_layers`| 8 |
|`hidden_states`| 32 |
|`dropout`| 0.2 |
|`epochs`| 30 |
|`batch_size`| 128 |
|`lr`| 0.01 |
|`factor`| 0.6 |
|`patience`| 1 |
|`optimizer`| `RAdam` |
|`scheduler`| `ReduceLROnPlateu`|

And we got the results:

<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/predictions_all.png" />
</p>

We can see that the trend if the predicted values, is similar to the original trend, and even that in the train and validation areas, we are giving pretty accurate prediciton.

Let's have a zoom in to the test prediction:

<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/predictions_test.png" />
</p>

Here we can see that the predicted trend behaves well, but after sometime it seems that we are losing resolution and diverging from the real stock values, although we are having success identifying sharp movements. We can also see how the fact that we've used moving average has smoothened our prediction, and it easy to observe how less spiky it is.

## Usage

To retrain the model run [stock_prediction_using_rwkv.ipynb](https://github.com/tomer9080/Stock-Prediction-Using-RWKV/stock_prediction_using_rwkv.ipynb). You can choose different stock to predict on in the relvant cell by just riplacing the ticker, and deciding on how much days you want to train (notice that different stocks has different number of data points). after you chose your hyperparameters, run all of the notebook and wait untill it's done.


## Files in the repository

| Folder |File name         | Purpose |
|------|----------------------|------|
|code|`stock_prediction_using_rwkv.ipynb`| Notebook which includes all data processing, training, and inference |
|images|`rwkv_arch.png`| Image that shows our arch including the RWKV model |
| |`data_set_split.png`| Image that shows our data split |
| |`predictions_all.png`| Image that shows the predictions obtained on all sets |
| |`predictions_test.png`| Image that shows our result on the test set |


## Further Work

The work we presented here achieved good results, but definitely there are aspects to improve and examine such as:
- Try running the model on a different stock.
- Examine adding Time2Vec embedding.
- Try and train the model on multiple stocks, and predict on them.

Hope this was helpful and please let us know if you have any comments on this work:

https://github.com/tomer9080

https://github.com/roilatzres