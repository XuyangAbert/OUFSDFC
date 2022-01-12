# Streaming feature selection framework using dyanmic density-based feature stream clustering (SFS-DFC)

# Description:
In this project, thirteen benchmark datasets are used:

Medical datasets: ALLAML, Lung, Arcene and Lymphoma.

Image datasets: Orlraws10P, Pixraws10P, WarpPIE10P, and COIL20.

Biological datasets: Colon, SMK, GLIMO, GLI-85, and Carcinom.

All these thirteen datasets can be found from the ASU feature selection repository using following link:

https://jundongl.github.io/scikit-feature/datasets.html

# Instructions for Running the code:

1. Install the following python packages first:

   numpy, scikit-learn, Orange3 (Python 3), and pandas.

2. For each folder, run the script "FC_test_stream.py" file to reproduce the results in the paper;
3. The chunk size is named as "Batchsize" variable and it can be changed in line 28 ;
4. Dataset name can be changed in line 27; 
5. For the statistical comparison using Friedman rank test and Nemenyi post-hoc test, go to folder "/ranking-test/" and run the "result.py" file;
