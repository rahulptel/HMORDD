#!/usr/bin/env
# coding: utf-8

# global vars
VENVS_DIR="/home/rahulpat/envs/"
VENV_NAME="hmordd"

# load module
echo "Load module..."
module purge
module load cuda
module load python/3.11
module load boost

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  # pip install
  echo "Install python packages..."
  pip install --no-index --upgrade pip
  pip install --no-index pymoo
  pip install --no-index numpy
  pip install --no-index matplotlib
  pip install --no-index scipy
  pip install --no-index scikit_learn
  pip install --no-index pandas
  pip install --no-index torch
  pip install --no-index pytorch_lightning
  pip install --no-index -U tensorboard
  pip install --no-index tensorboardX
  pip install --no-index pybind11
  pip install --no-index xgboost
  pip install --no-index hydra-core
# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""
