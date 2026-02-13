Setup
=====

Clone the repository where the developement is done::

   git clone git@github.com:mechmotum/gait-closed-loop-id.git

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

If we make updates in gait2d or opty, you will need to either update the
environment::

   conda deactivate
   conda env update -f gait-closed-loop-id-env.yml
   conda activate gait-closed-loop-id

or recreate the environment::

   conda deactivate
   conda env remove -n gait-closed-loop-id
   conda env create -f gait-closed-loop-id-env.yml
   conda activate gait-closed-loop-id

If you know that the only thing you need to update is gait2d you can do this to
update it::

   conda activate gait-closed-loop-id
   python -m pip install -U --no-deps --no-build-isolation -e git+https://github.com/csu-hmc/gait2d#egg=gait2d

Data
====

There is sample data in the ``data/`` directory for tracking but you can also
use data we collected by downloading these files:

- `sample calibration pose <https://drive.google.com/file/d/16BkXcR5F-7DsJNoXf9tjy5ujyN3pDMl0/view?usp=sharing>`_
- `sample gait cycles during perturbations <https://drive.google.com/file/d/1rsBbDih0fqa8v14fmY7__Ss01eaY7Ol9/view?usp=sharing>`_ [16Mb]

unzipping them and placing the two CSV files into the ``data/`` directory.

Filenames are:

- ``020-calibration-pose.csv``
- ``020-longitudinal-perturbation-gait-cycles.csv``

These are sample data from trial 20 from:

Moore JK, Hnat SK, van den Bogert AJ. 2015. An elaborate data set on human gait
and the effect of mechanical perturbations. PeerJ 3:e918
https://doi.org/10.7717/peerj.918

Run
===

Plot the calibration pose and a gait cycle::

   python src/utils.py

Solve an optimal control problem::

   python src/solve.py
