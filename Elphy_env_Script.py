"""
Electophysiology Script

Description:
    This script contains functions for downloading and analyzing electrophysiology data
    from the Allen Cell Types Database. In particular, it contains functions for
    downloading and analyzing data from the neuronal models.

Usage:
    You can run this script with the following command:
    python Elphy_env_Script.py

Author:
    Eleonora Fabbri

Date:
    September 25, 2023

"""

# #ATTENTION !!! : if I get an error ((the h5f files are unable to lock, whatever it means...), I need to set this environment variable BEFORE running this script
# import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Cleanup
from IPython import get_ipython

get_ipython().magic("reset -sf")

# Script for the new conda environment elphy_env
# I just dowloaded python=3.10 and NEURON with: pip3 install neuron
# Then I had to install pynwb with conda and allensdk with pip
import allensdk
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.api.queries.glif_api import GlifApi
from allensdk.model.biophys_sim.config import Config
import allensdk.core.json_utilities as json_utilities

# from allensdk.model.biophysical import utils
# from allensdk.model.biophysical.utils import Utils
# from allensdk.model.biophysical.utils import AllActiveUtils
# from allensdk.model.biophysical import runner
# from allensdk.model.biophysical.runner import load_description

import Organised_Script

import pynwb
from pynwb import NWBHDF5IO
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import pdb
from copy import deepcopy


#GLOBAL PARAMETERS

data_root_path = Path("/opt3/Eleonora/data")

#######################################################################################################################################################

# FUNCTIONS


def get_neur_id(specimen_id_list, neuronal_model_id_list, spec_id):
    """
    Retrieve the neuronal model ID associated with a specimen ID.

    Parameters:
    -----------
    specimen_id_list (list):
        List of specimen IDs.

    neuronal_model_id_list (list):
        List of neuronal model IDs corresponding to specimen IDs.

    spec_id (int):
        The specimen ID for which to retrieve the neuronal model ID.

    Returns:
    --------
    int or None:
        The neuronal model ID associated with the specified specimen ID, or None
        if the specimen ID is not found in the list.

    This function takes lists of specimen IDs (`specimen_id_list`) and their corresponding
    neuronal model IDs (`neuronal_model_id_list`) and retrieves the neuronal model ID
    associated with a specific specimen ID (`spec_id`).
    """

    index = specimen_id_list.index(spec_id)
    neur_id = neuronal_model_id_list[index]
    return neur_id


def get_sweep_data(
    nwb_file, sweep_number, time_scale=1e3, voltage_scale=1e3, stim_scale=1e12
):
    """
    Extract data and stim characteristics for a specific DC sweep from nwb file
    Parameters
    ----------
    nwb_file : string
        File name of a pre-existing NWB file.
    sweep_number : integer

    time_scale : float
        Convert to ms scale
    voltage_scale : float
        Convert to mV scale
    stim_scale : float
        Convert to pA scale

    Returns
    -------
    t : numpy array
        Sampled time points in ms
    v : numpy array
        Recorded voltage at the sampled time points in mV
    stim_start_time : float
        Stimulus start time in ms
    stim_end_time : float
        Stimulus end time in ms
    """

    nwb = NwbDataSet(nwb_file)
    sweep = nwb.get_sweep(sweep_number)
    stim = sweep["stimulus"] * stim_scale  # in pA
    stim_diff = np.diff(stim)
    stim_start = np.where(stim_diff != 0)[0][-2]
    stim_end = np.where(stim_diff != 0)[0][-1]

    # read v and t as numpy arrays
    v = sweep["response"] * voltage_scale  # in mV
    dt = time_scale / sweep["sampling_rate"]  # in ms
    num_samples = len(v)
    t = np.arange(num_samples) * dt
    stim_start_time = t[stim_start]
    stim_end_time = t[stim_end]
    return t, v, stim_start_time, stim_end_time


def flatten_json(json_obj, parent_key="", sep="_"):
    """
    Flatten a nested JSON object into a flat dictionary.

    Parameters:
    -----------
    json_obj (dict):
        The input JSON object to be flattened.

    parent_key (str, optional):
        The parent key for the current JSON object (used for recursion).

    sep (str, optional):
        The separator to be used when generating flattened keys.

    Returns:
    --------
    dict:
        A flattened dictionary where keys are composed of nested keys separated by `sep`.

    This function takes a nested JSON object (`json_obj`) and flattens it into a flat
    dictionary. Nested keys are combined using the specified `sep` character.
    """

    flattened = {}
    for key, value in json_obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key, sep=sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                flattened.update(
                    flatten_json({str(i): item}, f"{new_key}{sep}{key}", sep=sep)
                )
        else:
            flattened[new_key] = value
    return flattened


def get_json_parameters(specimen_id, cell_df):
    """
    Retrieve parameters from JSON files based on cell type and specimen ID.

    Parameters:
    -----------
    specimen_id (int):
        The specimen ID for which parameters are to be retrieved.

    cell_df (DataFrame):
        The cell DataFrame containing information about cell types.

    Returns:
    --------
    tuple:
        A tuple containing DataFrames for perisomatic, all_active, and GLIF parameters.

    This function takes a specimen ID and a cell DataFrame as input and retrieves parameters
    from JSON files based on the cell type. It returns a tuple of DataFrames containing
    the parameters for perisomatic, all_active, and GLIF cell types.

    If the cell_type is perisomatic or all_active, returns a DataFrame with the columns:
    'section','name,'value', 'mechanism' (taken from its fit.json file).
    If the cell_type is GLIF, it returns his neuron_config.json file as a DataFrame.
    
    Options: (I got them "manually" from the json files) 
     - section: 'somatic', 'basal', 'apical'
     - name: for all_active: 'g_pas', 'e_pas', 'cm', 'Ra', 'gbar_NaV', 'gbar_K_T', 'gbar_Kv2like',
             'gbar_Kv3_1', 'gbar_SK', 'gbar_Ca_HVA', 'gbar_Ca_LVA', 'gamma_CaDynamics',
                'decay_CaDynamics', 'gbar_Ih', 'gbar_Im_v2', 'gbar_Kd';
            for perisomatic: 'gbar_Im', 'gbar_Ih', 'gbar_NaTs', 'gbar_Nap', 'gbar_K_P',
             'gbar_K_T', 'gbar_SK', 'gbar_Kv3_1', 'gbar_Ca_HVA', 'gbar_Ca_LVA', 'gamma_CaDynamics',
                'decay_CaDynamics', 'g_pas', 'gbar_NaV', 'gbar_Kd', 'gbar_Kv2like', 'gbar_Im_v2';
     - mechanism: for all_active: '', 'NaV', 'K_T', 'Kd', 'Kv2like', 'Kv3_1', 'SK', 'Ca_HVA', 'Ca_LVA',
                    'CaDynamics', 'Ih', 'Im_v2'
                for perisomatic: '', 'Im', 'Ih', 'NaTs', 'Nap', 'K_P', 'K_T', 'SK', 'Kv3_1', 'Ca_HVA', 'Ca_LVA',
                    'CaDynamics', 'NaV', 'Kd', 'Kv2like', 'Im_v2'
     - GLIF neuron_config.json file parameters: El_reference, C, asc_amp_array, init_threshold, threshold_reset_method (params, name),
         th_inf, spike_cut_length, init_AScurrents, init_voltage, threshold_dynamics_method (params, name),
         voltage_reset_method (params, name), extrapolation_method_name, dt, voltage_dynamics_method (params, name),
         El, asc_tau_array, R_input, AScurrent_dynamics_method (params, name), AScurrent_reset_method (params, r, name),
         dt_multiplier, th_adapt, coeffs (a, C, b, G, th_inf, asc_amp_array), type.
    """

    specimen_id_str = str(specimen_id)
    row_idx = cell_df[cell_df.specimen__id == specimen_id].index[0]

    peri_param_df = pd.DataFrame()
    all_act_param_df = pd.DataFrame()
    glif_param_df = pd.DataFrame()

    if cell_df.loc[row_idx, "m__biophys_perisomatic"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/perisomatic")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "fit" in file:
                fit_json_path = base_dir_path / specimen_id_str / file
                with open(fit_json_path, "r") as file:
                    fit_json_data = json.load(file)
                section_list = []
                name_list = []
                value_list = []
                mechanism_list = []
                for item in fit_json_data["genome"]:
                    section_list.append(item["section"])
                    name_list.append(item["name"])
                    value_list.append(item["value"])
                    mechanism_list.append(item["mechanism"])

                peri_param_df = pd.DataFrame(
                    {
                        "section": section_list,
                        "name": name_list,
                        "value": value_list,
                        "mechanism": mechanism_list,
                    }
                )
    if cell_df.loc[row_idx, "m__biophys_all_active"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/all_active")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "fit" in file:
                fit_json_path = base_dir_path / specimen_id_str / file
                with open(fit_json_path, "r") as file:
                    fit_json_data = json.load(file)
                section_list = []
                name_list = []
                value_list = []
                mechanism_list = []
                for item in fit_json_data["genome"]:
                    section_list.append(item["section"])
                    name_list.append(item["name"])
                    value_list.append(item["value"])
                    mechanism_list.append(item["mechanism"])

                all_act_param_df = pd.DataFrame(
                    {
                        "section": section_list,
                        "name": name_list,
                        "value": value_list,
                        "mechanism": mechanism_list,
                    }
                )
    if cell_df.loc[row_idx, "m__glif"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/glif")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "config" in file:
                config_json_path = base_dir_path / specimen_id_str / file
                with open(config_json_path, "r") as file:
                    config_json_data = json.load(file)
                # Flatten the nested JSON data
                flat_data = flatten_json(config_json_data)
                # Create a DataFrame from the flattened data
                config_df = pd.DataFrame([flat_data])
                # Transpose the DataFrame for a more intuitive view
                config_df = config_df.T.reset_index()
                config_df.columns = ["Column", "Value"]
                glif_param_df = config_df
                # I need to remove the neuron_id from the Column names
                neur_id = get_neur_id(
                    specimen_id_list, neuronal_model_id_list, specimen_id
                )
                glif_param_df["Column"] = glif_param_df["Column"].str.replace(
                    (str(neur_id) + "_"), ""
                )
    # else:
    #     print('Error: cell_type must be perisomatic, all_active or GLIF')

    return peri_param_df, all_act_param_df, glif_param_df


def get_dict(specimen_id, cell_df):
    """
    Retrieve parameters as dictionaries based on cell type and specimen ID.

    Parameters:
    -----------
    specimen_id (int):
        The specimen ID for which parameters are to be retrieved.

    cell_df (DataFrame):
        The cell DataFrame containing information about cell types.

    Returns:
    --------
    tuple of dictionaries:
        A tuple containing dictionaries for perisomatic, all_active, and GLIF parameters.

    This function takes a specimen ID and a cell DataFrame as input and retrieves parameters
    as dictionaries based on the cell type. It returns a tuple of dictionaries containing
    the parameters for perisomatic, all_active, and GLIF cell types.
    If the cell_type is perisomatic or all_active, returns a dictionary with the keys = section_name
    and values = value (taken from its fit.json file)
    If the cell_type is GLIF, it returns his neuron_config.json file as a dict.
    (It's basically the same function as get_json_parameters, but it returns a dict instead of a DataFrame)
    """

    specimen_id_str = str(specimen_id)
    row_idx = cell_df[cell_df.specimen__id == specimen_id].index[0]
    peri_param_dict = {}
    all_act_param_dict = {}
    glif_param_dict = {}
    if cell_df.loc[row_idx, "m__biophys_perisomatic"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/perisomatic")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "fit" in file:
                fit_json_path = base_dir_path / specimen_id_str / file
                with open(fit_json_path, "r") as file:
                    fit_json_data = json.load(file)
                genome_data = fit_json_data["genome"]
                for entry in genome_data:
                    section_name = entry["section"]
                    field_name = entry["name"]
                    json_key = f"{section_name}_{field_name}"
                    value = entry["value"]
                    peri_param_dict[json_key] = value
    if cell_df.loc[row_idx, "m__biophys_all_active"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/all_active")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "fit" in file:
                fit_json_path = base_dir_path / specimen_id_str / file
                with open(fit_json_path, "r") as file:
                    fit_json_data = json.load(file)
                genome_data = fit_json_data["genome"]
                for entry in genome_data:
                    section_name = entry["section"]
                    field_name = entry["name"]
                    json_key = f"{section_name}_{field_name}"
                    value = entry["value"]
                    all_act_param_dict[json_key] = value
    if cell_df.loc[row_idx, "m__glif"] != 0:
        base_dir_path = Path(f"{str(data_root_path)}/Physiology/glif")
        file_in_cartella = os.listdir(base_dir_path / specimen_id_str)
        for file in file_in_cartella:
            if file.endswith(".json") and "config" in file:
                config_json_path = base_dir_path / specimen_id_str / file
                with open(config_json_path, "r") as file:
                    config_json_data = json.load(file)
                flat_data = flatten_json(config_json_data)
                neur_id = get_neur_id(
                    specimen_id_list, neuronal_model_id_list, specimen_id
                )
                for key, value in flat_data.items():
                    new_key = key.replace((str(neur_id) + "_"), "")
                    glif_param_dict[new_key] = value

    return peri_param_dict, all_act_param_dict, glif_param_dict


def merge_glif_df(specimens):
    """
    Merge GLIF parameters into a DataFrame for a list of specimens.

    Parameters:
    -----------
    specimens (list of int):
        List of specimen IDs for which GLIF parameters are to be merged.

    cell_df (DataFrame):
        The cell DataFrame containing information about cell types.

    Returns:
    --------
    DataFrame:
        A DataFrame containing merged GLIF parameters for the specified specimens.

    This function takes a list of specimen IDs and a cell DataFrame as input and merges
    GLIF parameters into a DataFrame for the specified specimens.
    """

    GLIF_df = pd.DataFrame()
    GLIF_df["specimen_id"] = specimens
    for specimen_id in specimens:
        peri_param_dict, all_act_param_dict, glif_param_dict = get_dict(
            specimen_id, cell_df
        )
        # for key in glif_param_dict.keys():
        for key, value in glif_param_dict.items():
            if key not in GLIF_df.columns:
                GLIF_df[key] = np.nan
            idx = GLIF_df[GLIF_df.specimen_id == specimen_id].index[0]
            GLIF_df.loc[idx, key] = value
    return GLIF_df



def merge_biophys_df(specimens, cell_df):
    """
    Retrieve parameters as dictionaries based on cell type and specimen ID.

    Parameters:
    -----------
    specimen_id (int):
        The specimen ID for which parameters are to be retrieved.

    cell_df (DataFrame):
        The cell DataFrame containing information about cell types.

    Returns:
    --------
    tuple of dictionaries:
        A tuple containing dictionaries for perisomatic, all_active, and GLIF parameters.

    This function takes a specimen ID and a cell DataFrame as input and retrieves parameters
    as dictionaries based on the cell type. It returns a tuple of dictionaries containing
    the parameters for perisomatic, all_active, and GLIF cell types.
    This function merges the glif_param_dict of multiple neuron, given in input as a list
    of their specimen_id, into a DataFrame.
    It's very similar to merge_glif_df, but it uses the function get_dict instead of get_json_parameters
    """

    PERI_df = pd.DataFrame()
    PERI_df["specimen_id"] = specimens
    ALL_ACT_df = pd.DataFrame()
    ALL_ACT_df["specimen_id"] = specimens
    for specimen_id in specimens:
        peri_param_dict, all_act_param_dict, glif_param_dict = get_dict(
            specimen_id, cell_df
        )
        for key, value in peri_param_dict.items():
            if key not in PERI_df.columns:
                PERI_df[key] = np.nan
            idx = PERI_df[PERI_df.specimen_id == specimen_id].index[0]
            PERI_df.loc[idx, key] = value
        for key, value in all_act_param_dict.items():
            if key not in ALL_ACT_df.columns:
                ALL_ACT_df[key] = np.nan
            idx = ALL_ACT_df[ALL_ACT_df.specimen_id == specimen_id].index[0]
            ALL_ACT_df.loc[idx, key] = value
    return PERI_df, ALL_ACT_df


#######################################################################################################################################################

# VARIABLE PARAMETERS


# I just followed the tutorial from the Allen Institute website, but I made some changes to the code to let it work
# I commented everything that it's not necessary.
ctc = CellTypesCache(manifest_file="cell_types/manifest.json")

specimen_id = 471819401 # It's just an example

# The directory 'download_dir' is the one where the model files will be saved, in my case it's the following one:
download_dir = "/opt3/Eleonora/scripts/messy_things/"  

# # a list of cell metadata for cells with reconstructions, download if necessary
cells = ctc.get_cells(require_reconstruction=True)

sweeps = ctc.get_ephys_sweeps(specimen_id)
sweep_numbers = defaultdict(list)
for sweep in sweeps:
    sweep_numbers[sweep["stimulus_name"]].append(sweep["sweep_number"])

# If I want to use the following methods, I need to make the commented imports work properly (because now they're messed up)
# sweep_numbers = data_set.get_sweep_numbers()
# sweep_number = sweep_numbers[0]
# sweep_data = data_set.get_sweep(sweep_number)


# I downloaded the query results from:
# http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20all%20active%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?q=model::NeuronalModel,rma::include,neuronal_model_template[name$eq%27Biophysical%20-%20perisomatic%27],rma::options[num_rows$eqall]
# and http://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCellTypesSpecimenDetail,rma::criteria,[m__glif$gt0] for the GLIF cells
# I took them from the Allen Institute Git repository (model-biophysical-passive_fitting-biophysical_archiver) and I saved them as json files

query_dir = f"{str(data_root_path)}"  # This is the directory where I saved the query results
with open(query_dir + "/query_perisomatic.json", "r") as file:
    perisomatic_data = json.load(file)
with open(query_dir + "/query_all_active.json", "r") as file:
    all_active_data = json.load(file)
with open(query_dir + "/query_glif.json", "r") as file:
    glif_data = json.load(file)


# This list contains the neuronal_models_ids of the cells that have biophysical perisoamtic models
msg_list = perisomatic_data["msg"]
msg_peri_df = pd.json_normalize(msg_list)
perisomatic_id_list = msg_peri_df.id

# This list contains the neuronal_models_ids of the cells that have biophysical all_active models
msg_list = all_active_data["msg"]
msg_all_act_df = pd.json_normalize(msg_list)
all_active_id_list = msg_all_act_df.id

# I am correcting the mainfest_file, so that it has also 'sweeps_by_type' a key; it should be done for every neuron I want to simulate
os.chdir(download_dir)
os.system("nrnivmodl modfiles/")
manifest_file = "manifest.json"
manifest_dict = json.load(open(manifest_file))

if "sweeps_by_type" not in manifest_dict["runs"][0]:
    manifest_dict["runs"][0]["sweeps_by_type"] = {
        "Long Square": sweep_numbers["Long Square"],
        "Short Square": sweep_numbers["Short Square"],
    }
json.dump(manifest_dict, open(manifest_file, "w"), indent=2)

# sweep_num = 43 # It's one of the sweeps of the Long Square

schema_legacy = dict(manifest_file=manifest_file)


# This is the turorial for a simulation loop, from the Allen Institute website (with some changes to make it work);
# the link to the tutorial is: http://alleninstitute.github.io/AllenSDK/biophysical_models.html.
# If I want to use the following methods, I need to make the commented imports work properly (because now they're messed up)

# my_args_dict={'manifest_file' : '/opt3/Eleonora/scripts/messy_things/manifest.json', 'axon_type' : 'truncated'}
# description = load_description(my_args_dict)
# my_utils = utils.create_utils(description)
# h = my_utils.h
# manifest = description.manifest
# morphology_path = manifest.get_path('MORPHOLOGY')
# my_utils.generate_morphology(morphology_path)
# stimulus_path = manifest.get_path('stimulus_path')
# nwb_out_path = manifest.get_path("output_path")  #I changed this line from the original simulation loop from the Allen example
# output = NwbDataSet(nwb_out_path)
# run_params = description.data['runs'][0]
# sweeps = run_params['sweeps']
# junction_potential = description.data['fitting'][0]['junction_potential']
# for sweep in sweeps:
#     my_utils.setup_iclamp(stimulus_path, sweep=sweep)
#     # configure stimulus
#     my_utils.setup_iclamp(stimulus_path, sweep=sweep)
#     # configure recording
#     vec = my_utils.record_values()
#     h.finitialize()
#     h.run()
#     # write to an NWB File
#     output_data = (np.array(vec['v']) - junction_potential) * mV
#     # try:
#     #     output.set_sweep(sweep, None, output_data)
#     # except:
#     #     pdb.set_trace()
#     output.set_sweep(sweep, None, output_data)

# t, v, stim_start, stim_end = get_sweep_data(stimulus_path, sweep_num)
# plt.plot(t, v)

# I am changing the directory to the one where the scripts are saved
scripts_dir = "/opt3/Eleonora/My_Scripts"
os.chdir(scripts_dir)


# I found out that it's possible to use the functions in AllenSDK.doc_template.examples_root.examples.simple.utils, that are much more quicker because it doesn't
# need to analize each sweep in a for loop, and get the same results as the simulation loop. 
# It's just used to run the Allen demo and to get the output of the simulation loop for plots.
# If I want to use the following methods, I need to make the commented imports work properly (because now they're messed up)

# from AllenSDK.doc_template.examples_root.examples.simple.utils import Utils

# other_utils = Utils(description)
# hh = other_utils.h
# manifest = description.manifest
# other_utils.generate_morphology()
# other_utils.setup_iclamp()
# vec = other_utils.record_values()
# hh.dt = 0.025
# hh.tstop = 20
# hh.finitialize()
# hh.run()
# mV = 1.0e-3
# ms = 1.0e-3
# output_data2 = np.array(vec['v']) * mV
# output_times2 = np.array(vec['t']) * ms
# output2 = np.column_stack((output_times2, output_data2))
# # write to a dat File
# v_out_path = manifest.get_path("output_path")
# with open (v_out_path, "w") as f:
#     np.savetxt(f, output2)


# I imported the df that contains the cell features from Organised_Scripts.py
cell_feat_orient_df = pd.read_csv(f"{str(data_root_path)}/cell_feat_orientation_data.csv")

cell_df = deepcopy(cell_feat_orient_df)
perisomatic_specimen_id = msg_peri_df.specimen_id
all_active_specimen_id = msg_all_act_df.specimen_id

# Get the specimen_id that have GLIF models
glif_api = GlifApi()
cells = glif_api.get_neuronal_models()
models = [nm for c in cells for nm in c["neuronal_models"]]
neuronal_model_id_list = []
specimen_id_list = []
for model in models:
    neuronal_model_id_list.append(model["id"])
    specimen_id_list.append(model["specimen_id"])

glif_specimen_id = specimen_id_list

id = 464198958
peri_param_df, all_act_param_df, glif_param_df = get_json_parameters(id, cell_df)


glif_specimens = [
    486111903,
    548459652,
    464198958,
    314900022,
    502999078,
    585947309,
    519749342,
    586071425,
    476131588,
    561517025,
]

GLIF_df = merge_glif_df(glif_specimens)

peri_specimens = [508279351, 464198958, 478499902, 314900022, 471786879]
all_act_specimens = [475202388, 471077857, 354190013, 485931158, 482690728]
