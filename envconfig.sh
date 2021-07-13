# install dependencies
#Should match Config.py
echo Initializing...

#Create directories to store persistent data
mkdir -p /workspace/conda
mkdir -p /workspace/data

#Create a new env called hugface
#conda update -y -n base -c defaults conda
export SHELL=/bin/bash
conda init bash
conda create --prefix /workspace/conda/hugface --clone base && \
#echo "conda init bash" >> ~/.bashrc && \
echo "conda activate /workspace/conda/hugface" >> ~/.bashrc && \
export PATH=/workspace/conda/hugface/bin:$PATH
echo "Test step 1"
source ~/.bashrc
source activate /workspace/conda/hugface
/bin/bash --login -c /workspace/mlgitpod/installing.sh
#export SHELL=/bin/bash
echo "Test step 2"
# conda activate /workspace/conda/hugface


# #Install conda packages for NLP higging face
# conda install -y pandas
# conda install -y scikit-learn 
# conda install -y matplotlib
# conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
# conda install -y -c conda-forge tensorflow
# conda install -y -c conda-forge keras
# conda install -y -c huggingface transformers
# conda install -y -c anaconda seaborn
# conda install -y -c conda-forge bokeh
# conda install -y -c plotly plotly

# #Install conda packages for to run jupyterlab
# # conda install -y -c conda-forge jupyterlab
# # conda install -y -c conda-forge beakerx
# # conda install -y -c conda-forge xeus-cling
# # conda install -y -c conda-forge xeus-python

# # Some extra packages for your environments
# # conda install -y -c anaconda pyodbc
# # conda install -y -c conda-forge xeus-cling
# # conda install -y -c conda-forge xeus-python
# # conda install -y -c conda-forge python-graphviz
# # conda install -y -c conda-forge tensorflow
# # conda install -y -c conda-forge keras
# # conda install -y -c conda-forge xtl
# # conda install -y -c conda-forge openblas
# # conda install -y -c conda-forge gdal
# # conda install -y -c conda-forge util-linux
# # conda install -y -c conda-forge libtiff
# # conda install -y -c conda-forge libgdal
# # conda install -y -c pytorch pytorch
# # conda install -y -c conda-forge dask
# # conda install -y -c conda-forge dash
# # conda install -y -c conda-forge dash-table
# # conda install -y -c conda-forge rx
# # conda install -y -c conda-forge dash-core-components
# # conda install -y -c conda-forge cassandra-driver
# # conda install -y -c ranaroussi yfinance

# #Notes
# #To run jupyter lab : jupyter lab --ip=0.0.0.0 --allow-root
# echo Done...
