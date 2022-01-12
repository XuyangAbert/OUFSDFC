# Streaming feature selection framework using dyanmic density-based feature stream clustering (SFS-DFC)

# Description:
In this project, thirteen benchmark datasets from ASU Feature Selection Repository [1] are used:

Medical datasets: ALLAML, Lung, Arcene and Lymphoma.

Image datasets: Orlraws10P, Pixraws10P, WarpPIE10P, and COIL20.

Biological datasets: Colon, SMK, GLIMO, GLI-85, and Carcinom.

# Instructions for Running the code:

1. Install the following python package first:

   numpy, scikit-learn, Orange3 (Python 3), and pandas

3. For each folder, run the script "FC_test_stream.py" file to reproduce the results in the paper
4. The chunk size is named as "Batchsize" variable and it can be changed in line 28 
5. Dataset name can be changed in line 27 
6. For the statistical comparison using Friedman rank test and Nemenyi post-hoc test, go to folder "/ranking-test/" and run the "result.py" file.
