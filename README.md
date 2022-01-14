# Streaming feature selection framework using dyanmic density-based feature stream clustering (SFS-DFC)

# Description:
The "Supplemental-Results.pdf" file shows the number of selected features for all seven compared SFS methods and the proposed SFS-DFC method. Some discussions are provided as well.

In this project, thirteen benchmark datasets are used:

Medical datasets: ALLAML, Lung, Arcene and Lymphoma.

Image datasets: Orlraws10P, Pixraws10P, WarpPIE10P, and COIL20.

Biological datasets: Colon, SMK, GLIMO, GLI-85, and Carcinom.

All these thirteen datasets can be found from the ASU feature selection repository using following link:

https://jundongl.github.io/scikit-feature/datasets.html

# Instructions for Running the code for SFS-DC framework:

1. Install the following python packages first:

   numpy, scikit-learn, Orange3 (Python 3), and pandas.

2. For the proposed SFS-DFC method, go to the directory "Experiments/Codes/XXX/" to find the correspdong folder for each dataset and run the script "FC_test_stream.py" file to reproduce the results in the paper;
3. The chunk size is named as "Batchsize" variable and it can be changed in line 28 ;
4. Dataset name can be changed in line 27; 
5. For the statistical comparison using Friedman rank test and Nemenyi post-hoc test, go to folder "/Ranking_test/" and run the "result.py" file;
