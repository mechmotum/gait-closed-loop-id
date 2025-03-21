Setup
=====

Clone the two repositories where the developement is done::

   git clone git@github.com:mechmotum/gait-closed-loop-id.git
   git clone git@github.com:csu-hmc/gait2d.git

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

Change back into the project repository::

   cd ../gait-closed-loop-id

Run
===

Run the code that evaluates the differential equations::

   python src/evaluate.py

Solve an optimal control problem::

   python src/solve.py
