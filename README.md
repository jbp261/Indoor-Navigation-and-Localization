# Indoor Navigation and Localization

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [mlxtend](https://rasbt.github.io/mlxtend/)
- [keres](https://keras.io/)
- [seaborn](https://seaborn.pydata.org/index.html)
- [H5py](https://www.h5py.org/)
- [pydot](https://pypi.org/project/pydot/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### Code

The complete code is provided in the `boston_housing.ipynb` notebook file. The following files are necessory to run `boston_housing.ipynb`
- `Task.py`
- `Agent.py`
- `ReplayBuffer.py`
- `QNetwork.py`
- `VAE.py`
- `VAE_action.py`
- `environment.py`

### Run

In a terminal or command window, navigate to the top-level project directory `Indoor Navigation and Localization/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Navigation_Project.ipynb\
```  
or
```bash
jupyter notebook Navigation_Project.ipynb\
```

This will open the Jupyter Notebook software and project file in your browser.\

### Data

There are two datasets. One is iBeacon_RSSI_labeled, and the other one is iBeacon_RSSI_unlabeled. The datasets are available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/BLE+RSSI+Dataset+for+Indoor+localization+and+Navigation).\

**Features**
1.  `location`: The location of receiving RSSIs from ibeacons b3001 to b3013; symbolic values showing the column and row of the location on the map (e.g., A01 stands for column A, row 1). 
2. `Date`: Datetime in the format of dd-mm-yyyy hh : mm : yyyy 
3. `b3001 - b3013`: RSSI readings corresponding to the iBeacons; numeric, integers only. 
}
