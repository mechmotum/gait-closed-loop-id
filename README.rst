Setup
=====

Clone the two repositories where the developement is done::

   git clone git@github.com:mechmotum/gait-closed-loop-id.git
   git clone git@github.com:csu-hmc/gait2d.git

If using Spyder, install it along with ``spyder-kernels`` in your base
environment::

   conda install spyder spyder-kernels

Install the conda environment for the gait closed loop id project::

   cd gait-closed-loop-id
   conda env create -f gait-closed-loop-id-env.yml
   conda activate gait-closed-loop-id

You will also need a working C compiler on your operating system. For Windows,
you'll need the right compiler for the Python version you are using. See
https://wiki.python.org/moin/WindowsCompilers for more info.

Install an editable development version of gait2d::

   cd ../gait2d
   python -m pip install --no-dependencies --no-build-isolation --editable  .

If that fails, try::

   python -m pip install --no-deps --no-build-isolation git+https://github.com/csu-hmc/gait2d#egg=gait2d

Change back into the project repository::

   cd ../gait-closed-loop-id

Data
====

There is sample data in the ``data/`` directory for tracking but you can use
data we collected by downloading this `sample file
<https://drive.google.com/file/d/1rsBbDih0fqa8v14fmY7__Ss01eaY7Ol9/view?usp=sharing>`_
[16Mb], unzipping it, and placing the CSV file into the ``data/`` directory.
This is sample data from trial 20 from:

Moore JK, Hnat SK, van den Bogert AJ. 2015. An elaborate data set on human gait
and the effect of mechanical perturbations. PeerJ 3:e918
https://doi.org/10.7717/peerj.918

Run
===

Run the code that evaluates the differential equations::

   python src/evaluate.py

Solve an optimal control problem::

   python src/solve.py
