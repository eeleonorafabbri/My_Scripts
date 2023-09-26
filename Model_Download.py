"""
Model Download Script

Description:
    This script downloads the biophysical models (perisomatic and all_active) and the glif models, in /data/Physiology.
    For each neuron, I am creating a folder with the specimen_id as name and I am saving the data there, which are two folders (modfiles and work),
    two json files (manifest and fit_parameters), an nwb file, and two swc file (marker and "normal" morphology data)

Usage:
    You can run this script with the following command:
    python Model_Download.py 

Author:
    Eleonora Fabbri

Date:
    September 25, 2023

"""


from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.simulate_neuron import simulate_neuron


import os
from pathlib import Path
import pandas as pd
import json

# I downloaded the query results from:
# http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20all%20active%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20perisomatic%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCellTypesSpecimenDetail,rma::criteria,[m__glif$gt0] for the GLIF cells
# I took them from the Allen Institute Git repository (model-biophysical-passive_fitting-biophysical_archiver) and I saved them as json files

query_dir = "/opt3/Eleonora/data"  # This is the directory where I saved the query results
with open(query_dir + "/query_perisomatic.json", "r") as file:
    perisomatic_data = json.load(file)
with open(query_dir + "/query_all_active.json", "r") as file:
    all_active_data = json.load(file)
with open(query_dir + "/query_glif.json", "r") as file:
    glif_data = json.load(file)

msg_list = perisomatic_data["msg"]
msg_peri_df = pd.json_normalize(msg_list)
perisomatic_id_list = msg_peri_df.id

msg_list = all_active_data["msg"]
msg_all_act_df = pd.json_normalize(msg_list)
all_active_id_list = msg_all_act_df.id


download_folder_dir = Path(
    "/opt3/Eleonora/data/Physiology/"
)  # This is the directory of the main folder where I want to save the data
perisomatic_dir = download_folder_dir / "perisomatic"
all_active_dir = download_folder_dir / "all_active"
glif_dir = download_folder_dir / "glif"


# Download perisomatic and all_active data:

bp = BiophysicalApi()
bp.cache_stimulus = True

for neur_id in perisomatic_id_list:
    specimen_id = msg_peri_df.loc[msg_peri_df["id"] == neur_id, "specimen_id"].iloc[0]
    neuron_directory = Path(perisomatic_dir) / str(specimen_id)
    if not neuron_directory.exists():
        neuron_directory.mkdir(parents=True, exist_ok=True)
    # if in the folder there less than 7 items, then download the data
    if len(os.listdir(neuron_directory)) < 7:
        bp.cache_data(neur_id, working_directory=neuron_directory)
    else:
        print(f"Neuron {neur_id} already downloaded")

for neur_id in all_active_id_list:
    specimen_id = msg_all_act_df.loc[
        msg_all_act_df["id"] == neur_id, "specimen_id"
    ].iloc[0]
    neuron_directory = Path(all_active_dir) / str(specimen_id)
    if not neuron_directory.exists():
        neuron_directory.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(neuron_directory)) < 7:
        bp.cache_data(neur_id, working_directory=neuron_directory)
    else:
        print(f"Neuron {neur_id} already downloaded")


# Download glif data:

glif_api = GlifApi()
cells = glif_api.get_neuronal_models()
models = [nm for c in cells for nm in c["neuronal_models"]]
neuronal_model_id_list = []
specimen_id_list = []
for model in models:
    neuronal_model_id_list.append(model["id"])
    specimen_id_list.append(model["specimen_id"])


# Iterate over both lists simultaneously using zip
for neuronal_model_id, specimen_id in zip(neuronal_model_id_list, specimen_id_list):
    neuron_folder = os.path.join(glif_dir, str(specimen_id))
    os.makedirs(neuron_folder, exist_ok=True)
    num_elements_in_folder = len(os.listdir(neuron_folder))
    # Download data only if there are fewer than 3 elements in the folder
    if num_elements_in_folder < 3:
        # Download model metadata
        glif_api = GlifApi()
        nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]
        # Download the model configuration file
        nc = glif_api.get_neuron_configs([neuronal_model_id])[neuronal_model_id]
        neuron_config = glif_api.get_neuron_configs([neuronal_model_id])
        json_utilities.write(
            os.path.join(neuron_folder, "neuron_config.json"), neuron_config
        )
        # Download information about the cell
        ctc = CellTypesCache()
        ctc.get_ephys_data(
            nm["specimen_id"], file_name=os.path.join(neuron_folder, "stimulus.nwb")
        )
        ctc.get_ephys_sweeps(
            nm["specimen_id"],
            file_name=os.path.join(neuron_folder, "ephys_sweeps.json"),
        )
    else:
        print(f"Neuron {neuronal_model_id} already downloaded")


# There're a lot of cells that have glif data, so I didn't downloaded all of them, but I started from the cells 
# on which I had morphological and electrophysiological data (that are contained in the cell_feat_orientation_data.csv file, created in 
# the Organised_Script.py script).

cell_feat_orient_df = pd.read_csv("/opt3/Eleonora/data/cell_feat_orientation_data.csv")

glif_common_id = set(cell_feat_orient_df.specimen__id) & set(specimen_id_list)
glif_common_id = [int(id) for id in glif_common_id]

linked_elements = [
    neuronal_model_id_list[specimen_id_list.index(id)] for id in glif_common_id
]
for neuronal_model_id, specimen_id in zip(linked_elements, glif_common_id):
    neuron_folder = os.path.join(glif_dir, str(specimen_id))
    os.makedirs(neuron_folder, exist_ok=True)
    num_elements_in_folder = len(os.listdir(neuron_folder))
    # Download data only if there are fewer than 3 elements in the folder
    if num_elements_in_folder < 3:
        # Download model metadata
        glif_api = GlifApi()
        nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]
        # Download the model configuration file
        nc = glif_api.get_neuron_configs([neuronal_model_id])[neuronal_model_id]
        neuron_config = glif_api.get_neuron_configs([neuronal_model_id])
        json_utilities.write(
            os.path.join(neuron_folder, "neuron_config.json"), neuron_config
        )
        # Download information about the cell
        ctc = CellTypesCache()
        ctc.get_ephys_data(
            nm["specimen_id"], file_name=os.path.join(neuron_folder, "stimulus.nwb")
        )
        ctc.get_ephys_sweeps(
            nm["specimen_id"],
            file_name=os.path.join(neuron_folder, "ephys_sweeps.json"),
        )
    else:
        print(f"Neuron {neuronal_model_id} already downloaded")