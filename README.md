# Over-Parametrized
Repository for my final project for the <a href="http://klab.tch.harvard.edu/academia/classes/BAI/notes_bai.html#sthash.tNbkGizB.DDZFHugz.dpbs" target="_top">Biological and Artificial Intelligence class</a> (Neuro 140) at Harvard, taught by Professor Gabriel Kreiman.

## Modern machine learning lives in the over-parametrized regime
Modern architecture, such as deep neural networks, typically have many more free parameters than examples used to train them.
Classical statistics predict they would over-fit the training set, with poor generalisation performance outside on a test sample.
Why this is not the case is still an open question, that I investigate here from the perspective of kernel machines.

## Conda environment
The simplest wat to run this code is to create a Conda environment from the <a href="./environment.yml" target="_top">environment file</a>, by simply running `conda env create -f environment.yml`.
If you are not familiar with Conda environments you can have a look <a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" target="_top">here</a> and learn more! They are extremelly powerful!

## The condition number for random matrices
The first piece of the project is to study the shape of the condition number for linear system and kernel machines.
In <a href="./Condition_number.ipynb" target="_top">Condition number notebook</a>, you can compute the condition number for random matrices associated to linear system and kernel machines, by varying the number of parameters and the dimensionality of the data.
The condition number has a double descent shape, with a peak when the number of training examples (or conditions, for a linear system) approach the dimensionality of the data.

## Classification of hand-written MNIST digits with support-vector machines
To observe the effect of over-parametrization, I looked for the double descent pattern in the test error of a classifier, trained to distinguish the notorious MNIST handwritten digits.

For running <a href="./MNIST_SVM_classification.ipynb" target="_top">the MNIST SVM notebook</a> you will need to download the MNIST train and test datasets in csv from the <a href="https://www.kaggle.com/oddrationale/mnist-in-csv" target="_top">Kaggle website</a> and place them in a folder called `data`.
The notebook allows to change many different parameters and produce test error curves as a function of the number of training examples. The function `analytic_pipeline_v` can be call with an array of parameters that are varied while all the rest is kept at default values.

## Kernel ridge-less regression with kernel machines, implemented with a closed-form expression
Another study implements a kernel ridge-less (without regularisation) regression using a closed-form expression for kernel machines.
The classes in <a href="./lib/kernel_utils.py" target="_top">the main class definition file</a> can be customised to handle different type of data.
I adapt them to handle <a href="./Analytic_kernel_regression.ipynb.ipynb" target="_top">synthetic data</a>, with which you could see beautiful double descent pattern, and <a href="./MNIST_kernel_regression.ipynb" target="_top">MNIST handwritten digits</a>.

## Reach out to me for collaborating and contributing!
I'd love to share this project and work together with other people!
