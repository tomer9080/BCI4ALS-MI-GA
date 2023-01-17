## OptimalTransportOnSPD

The code in this repository creates the figures presented in this article: https://arxiv.org/abs/1906.00616
Please notice that in order to apply the code to the data sets, they need to be downloaded first from the following specified links.
The code was developed and tested in Matlab R2019a.

# Toy problem (Figure 2):
* For the toy examples we use the OT solver in the following link: http://www.numerical-tours.com/matlab/optimaltransp_1_linprog/#30
* The toy example can be fully run using the two scripts in the ToyExample folder.
* The script ‘MainPlotCone.m’ generates figures (a) and (b) in Figure 2 in the paper.
* The script ‘MainPlotPlan.m’ generates figures (c) and (d) in Figure 2 in the paper.

# Motor imagery task (Figure 4):
* Download the data from the following link (data set 2a): 
  *.gdf files are downloaded from here (you need to subscribe first): http://www.bbci.de/competition/iv/#dataset2a
  *.mat files are downloaded from here (link that says 'labels'): http://www.bbci.de/competition/iv/results/index.html#dataset2a
  you can find its description here: http://www.bbci.de/competition/iv/desc_2a.pdf.
* The data files are read using BioSig. You can download it from here: http://biosig.sourceforge.net.
  Please note that you need to INSTALL the package before using it.
  For matlab users, you need to run the install.m file in 'biosig' folder.
  Before installing define the parameter 'BIOSIG_DIR' according to the relevant path on your computer.
* The ‘GetEvents.m’ script (function) reads the data and applies a simple preprocessing to it as reported in the
  previous works by Barachant et al. (2012); Zanini et al. (2018); Yair et al. (2019).
  Before using it you need to update the 'dirPath' parameter to the dataset folder.
* The ‘MainBciPlot.m’ script applies the proposed algorithm for domain adaptation to the data. The script plots 
  the t-SNE representation before and after the adaptation similarly to the figures in the paper and trains a classifier accordingly.

# Event related potential P300 task:
* Download the data from the following link: https://zenodo.org/record/2669187#.XyqfjigzaUk
  you can find its description here: https://arxiv.org/abs/1905.05182.
* The data files are read and saved in ‘.mat’ format using the python code in the py.BI.EEG.2013-GIPSA repository.
  You can download it from here: https://github.com/plcrodrigues/py.BI.EEG.2013-GIPSA.
* The ’MainErpPlot.m’ script applies the proposed algorithm for domain adaptation to the data. The script plots
  the t-SNE representation before and after the adaptation similarly to the figures in the paper and trains a classifier
  accordingly.
* For computing the appended covariance features, we used this P300 detector:
  https://www.epfl.ch/labs/mmspg/research/page-58317-en-html/bci-2/bci_datasets/
  One can download it and replace line 23 in the script.
* In order to create the 'Subject' mat files, one should add the following code to the file 'classification_scores.py' in GIPSA:
  * import scipy.io as sio
  * sio.savemat('Subject' + str(subject) + 'Session' + str(session) + '.mat', {'mX':X, 'vY':y})
