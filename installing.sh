
#Install conda packages for NLP higging face
conda install -y pandas
conda install -y scikit-learn 
conda install -y matplotlib
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y -c conda-forge tensorflow
conda install -y -c conda-forge keras
conda install -y -c huggingface transformers
conda install -y -c anaconda seaborn
conda install -y -c conda-forge bokeh
conda install -y -c plotly plotly
conda install -y -c huggingface -c conda-forge datasets
conda install -y -c conda-forge sentencepiece


#Install conda packages for to run jupyterlab
# conda install -y -c conda-forge jupyterlab
# conda install -y -c conda-forge beakerx
# conda install -y -c conda-forge xeus-cling
# conda install -y -c conda-forge xeus-python

# Some extra packages for your environments
# conda install -y -c anaconda pyodbc
# conda install -y -c conda-forge xeus-cling
# conda install -y -c conda-forge xeus-python
# conda install -y -c conda-forge python-graphviz
# conda install -y -c conda-forge tensorflow
# conda install -y -c conda-forge keras
# conda install -y -c conda-forge xtl
# conda install -y -c conda-forge openblas
# conda install -y -c conda-forge gdal
# conda install -y -c conda-forge util-linux
# conda install -y -c conda-forge libtiff
# conda install -y -c conda-forge libgdal
# conda install -y -c pytorch pytorch
# conda install -y -c conda-forge dask
# conda install -y -c conda-forge dash
# conda install -y -c conda-forge dash-table
# conda install -y -c conda-forge rx
# conda install -y -c conda-forge dash-core-components
# conda install -y -c conda-forge cassandra-driver
# conda install -y -c ranaroussi yfinance

#Notes
#To run jupyter lab : jupyter lab --ip=0.0.0.0 --allow-root
echo Done...
