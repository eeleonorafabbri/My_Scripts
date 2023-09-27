# My_Scripts
The scripts in this repo are about analysis and methods on the Morphology and Electophysiology of the Allen Institute's CellTypes cells.
In detail:
- the Organised_Script.py contains all the functions created to analyse the Allen Cell Types data, in particular the electrophysiological and morphological features of the cells, that can be easily read in DataFrames;
  it should be run as the first script, because it contains all the necessary data
- the Viz.py contains methods for visualizing the morphology of neurons; in the Organised_Script.py I've instantiating the Viz class (taken from the Viz.py script) and made some examples of the methods that can be used
- the Model_Download.py downloads the biophysical models (perisomatic and all_active) and the glif models, so it's not necessary if you have already downloaded the data
- the Elphy_env_Script.py contains functions for downloading and analyzing data from the neuronal models; it should be run once you have downloaded the models from Model_Download.py script or somewhere else
