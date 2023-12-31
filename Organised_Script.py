"""
Organised Script

Description:
    This script contains all the functions that I have created to analyse the Allen Cell Types data,
    in particular the electrophysiological and morphological features of the cells.

Usage:
    You can run this script with the following command:
    python Organised_Script.py

Author:
    Eleonora Fabbri

Date:
    September 25, 2023

"""


# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve, newton, bisect

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import ReporterStatus as RS

from Viz import Viz

# builtins
from pathlib import Path
import time
import pprint
import pdb
from copy import deepcopy
import math
from math import sqrt

# GLOBAL PARAMETERS

data_root_path = Path("/opt3/Eleonora/data")
output_dir = Path(f"{str(data_root_path)}/Morphology")
ctc = CellTypesCache(manifest_file=output_dir / "manifest.json")
axon_color = "blue"
bas_dendrite_color = "red"
api_dendrite_color = "orange"


# FUNCTIONS


def get_features_df():
    """
    This function returns 3 different DataFrames (taken from the Allen, like
    using ctc and methods):
    - ef_df: dataframe containing all the electrophysiological features of the cells
    - mor_df: dataframe containing all the morphological features of the cells
    - feat_df: dataframe containing all the features of the cells
    """
    # feat_df = ctc.get_all_features(dataframe=True)
    feat_df = pd.read_csv(f"{str(data_root_path)}/All_features_cells.csv")
    ef_df = pd.read_csv(f"{str(data_root_path)}/ef_data.csv")
    mor_df = pd.read_csv(f"{str(data_root_path)}/mor_data.csv")
    return ef_df, mor_df, feat_df


def reconstruct(id_cell):
    """
    Returns the morphology (the swc file) of the cell with its specimen_id, in which
    there are id, type, the coordinates of each node of the cell, r and parent_id
    """
    morphology = ctc.get_reconstruction(id_cell)
    return morphology



def axon_or_dendrite(morph_df):
    """
    Return the indices of axons, basal dendrites, and apical dendrites in the given DataFrame.

    Parameters:
    -----------
    morph_df : pandas.DataFrame
        A DataFrame containing morphological data with a "type" column
        ( created with:
        morph = reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)   )

    Returns:
    --------
    axons_idx : pandas.Int64Index
        An Int64Index containing indices corresponding to axons in the DataFrame.

    basal_dendrite_idx : pandas.Int64Index
        An Int64Index containing indices corresponding to basal dendrites in the DataFrame.

    apical_dendrite_idx : pandas.Int64Index
        An Int64Index containing indices corresponding to apical dendrites in the DataFrame.

    Notes:
    ------
    This function filters morphological data based on the values in the "type" column,
    identifying axons (type 2), basal dendrites (type 3), and apical dendrites (type 4).
    It returns three separate Int64Index objects, one for each category.

    Example:
    --------
    # Creating a DataFrame containing morphological data for a specific neuron, using its specimen_id
    >>> cell_id = 479013100
    >>> morph = reconstruct(cell_id)
    >>> morph_df = pd.DataFrame(morph.compartment_list)

    # Call the function to get indices of axons, basal dendrites, and apical dendrites
    >>> axon_indices, basal_dendrite_indices, apical_dendrite_indices = axon_or_dendrite(morph_df)

    >>> print("Axon indices:", axon_indices)
    Axon indices: Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64')

    >>> print("Basal dendrite indices:", basal_dendrite_indices)
    Basal dendrite indices: Int64Index([ 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
            ...
            976, 977, 978, 979, 980, 981, 982, 983, 984, 985],
           dtype='int64', length=970)

    >>> print("Apical dendrite indices:", apical_dendrite_indices)
    Apical dendrite indices: Int64Index([ 986,  987,  988,  989,  990,  991,  992,  993,  994,  995,
            ...
            1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812],
           dtype='int64', length=827)

    """

    axons_idx = morph_df[morph_df["type"].values == 2].index
    basal_dendrite_idx = morph_df[morph_df["type"].values == 3].index
    apical_dendrite_idx = morph_df[morph_df["type"].values == 4].index
    return axons_idx, basal_dendrite_idx, apical_dendrite_idx


def correct_slice_angle(alpha, x, y):
    """
    Rotate a 2D point (x, y) by an angle alpha around the origin.

    Parameters:
    -----------
    alpha : float
        The angle in degrees by which to rotate the point.

    x : float
        The x-coordinate of the point to be rotated.

    y : float
        The y-coordinate of the point to be rotated.

    Returns:
    --------
    x_new : float
        The new x-coordinate of the rotated point.

    y_new : float
        The new y-coordinate of the rotated point.

    Notes:
    ------
    This function performs a simple 2D rotation of a point (x, y) around the origin
    by an angle alpha in degrees. It uses the standard rotation formula.

    Example:
    --------

    # Rotate a point (2, 3) by 45 degrees
    >>> x, y = 2, 3
    >>> rotated_x, rotated_y = correct_slice_angle(45, x, y)

    >>> print("Original point:", (x, y))
    Original point: (2, 3)

    >>> print("Rotated point:", (rotated_x, rotated_y))
    Rotated point: (3.5355339059327378, 0.7071067811865479)

    """

    alpha = math.radians(alpha)
    x_new = x * (math.cos(alpha)) + y * (math.sin(alpha))
    y_new = -x * (math.sin(alpha)) + y * (math.cos(alpha))
    return x_new, y_new


def proper_rotation(slice_angle, upright_angle, x1, y1, z1, shrink_factor):
    """
    Apply a proper 3D rotation to a point (x1, y1, z1).

    This function performs a series of transformations on a 3D point, starting with
    a rotation of angle `slice_angle` around the x-axis, followed by a rotation
    of angle `upright_angle` around the z-axis. Before the rotations, the z-coordinate
    is scaled by a factor `shrink_factor`.

    ATTENTION: I have the data about angles only for mice cells, not for the human ones!

    Parameters:
    -----------
    slice_angle : float
        The angle in degrees for the rotation around the x-axis.
        It can be found in the cell_feat_orient_df with the name "estimated_slice_angle"

    upright_angle : float
        The angle in degrees for the rotation around the z-axis.
        It can be found in the cell_feat_orient_df with the name "upright_angle"

    x1 : float
        The original x-coordinate of the point.

    y1 : float
        The original y-coordinate of the point.

    z1 : float
        The original z-coordinate of the point.

    shrink_factor : float
        The scaling factor to apply to the z-coordinate before rotation.
        It can be found in the cell_feat_orient_df with the name "estimated_shrinkage_factor"

    Returns:
    --------
    x3 : float
        The final x-coordinate of the rotated and transformed point.

    y3 : float
        The final y-coordinate of the rotated and transformed point.

    z3 : float
        The final z-coordinate of the rotated and transformed point.

    Notes:
    ------
    This function applies a proper 3D rotation sequence to a point, which includes
    a scaling, followed by two rotations. It uses standard rotation formulas.

    Example:
    --------

    # Rotate a point (1, 0, 0) with specific parameters
    >>> x, y, z = 1, 0, 0
    >>> slice_angle = 30  # degrees
    >>> upright_angle = 45  # degrees
    >>> shrink_factor = 0.5

    >>> rotated_x, rotated_y, rotated_z = proper_rotation(slice_angle, upright_angle, x, y, z, shrink_factor)

    >>> print("Original point:", (x, y, z))
    Original point: (1, 0, 0)

    >>> print("Rotated point:", (rotated_x, rotated_y, rotated_z))
    Rotated point: (0.7071067811865476, 0.6123724356957945, 0.3535533905932737)

    """

    slice_angle = math.radians(slice_angle)
    upright_angle = math.radians(upright_angle)
    z1 = z1 * shrink_factor
    x2 = x1 * (math.cos(upright_angle)) - y1 * (math.sin(upright_angle))
    y2 = x1 * (math.sin(upright_angle)) + y1 * (math.cos(upright_angle))
    z2 = z1
    x3 = x2
    y3 = y2 * (math.cos(slice_angle)) - z2 * (math.sin(slice_angle))
    z3 = y2 * (math.sin(slice_angle)) + z2 * (math.cos(slice_angle))
    return x3, y3, z3


def _find_max_eucl_distance(cell_id, cell_feat_orient_new_df):
    """
    This function was created just to inspect how was alculated the max_eucl_distance in the Allen DataFrames
    (so it's useless).

    Calculate various maximum Euclidean distances and indices related to a cell's morphology.

    This function calculates several maximum Euclidean distances and indices based on a cell's
    morphology data, including:

    1. Maximum Euclidean distance between the soma and the farthest node of the cell.
    2. Maximum Euclidean distance between the soma and the farthest node of the cell after rotation.
    3. Maximum Euclidean distance between the soma and the farthest node of the cell after applying
        shrinkage.
    4. Maximum distance between the soma and the farthest node of the cell in the XY plane.
    5. Maximum distance between the soma and the farthest node of the cell after rotation in the XY plane.
    6. Index of the farthest node of the cell after rotation in the XY plane (commented out).

    Parameters:
    -----------
    cell_id : int
        The unique identifier of the cell.

    cell_feat_orient_new_df : pandas.DataFrame
        A DataFrame containing cell features including orientation-related data.

    Returns:
    --------
    max_eucl_distance : float
        The maximum Euclidean distance between the soma and the farthest node of the cell.

    max_rot_eucl_distance : float
        The maximum Euclidean distance between the soma and the farthest node of the cell
        after rotation.

    max_shrinked_eucl_distance : float
        The maximum Euclidean distance between the soma and the farthest node of the cell
        after shrinkage.

    max_xy_distance : float
        The maximum distance between the soma and the farthest node of the cell in the XY plane.

    max_xy_rot_distance : float
        The maximum distance between the soma and the farthest node of the cell after rotation
        in the XY plane.

    # idx_position_far_node : int
    #     The index of the farthest node of the cell after rotation in the XY plane.

    Notes:
    ------
    This function calculates various morphological distances and indices based on the cell's
    reconstruction data and orientation-related information. It uses the `reconstruct` and
    `proper_rotation` functions from the provided module.

    """

    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    x_soma = morph_df.loc[0, "x"]
    y_soma = morph_df.loc[0, "y"]
    z_soma = morph_df.loc[0, "z"]
    cell_idx = cell_feat_orient_new_df[
        cell_feat_orient_new_df["specimen_id"] == cell_id
    ].index
    slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
    upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
    shrink_factor = cell_feat_orient_new_df.loc[cell_idx, "estimated_shrinkage_factor"]
    z_soma_shrink = z_soma * shrink_factor
    x_soma_rot, y_soma_rot, z_soma_rot = proper_rotation(
        slice_angle, upright_angle, x_soma, y_soma, z_soma, shrink_factor
    )
    eucl_distance = []
    shrinked_eucl_distance = []
    rot_eucl_distance = []
    xy_distance = []
    xy_rot_distance = []
    for idx in morph_df.index:
        x_node = morph_df.loc[idx, "x"]
        y_node = morph_df.loc[idx, "y"]
        z_node = morph_df.loc[idx, "z"]
        eucl_distance.append(
            sqrt(
                (x_node - x_soma) ** 2 + (y_node - y_soma) ** 2 + (z_node - z_soma) ** 2
            )
        )
        z_node_shrink = z_node * shrink_factor
        shrinked_eucl_distance.append(
            sqrt(
                (x_node - x_soma) ** 2
                + (y_node - y_soma) ** 2
                + (z_node_shrink - z_soma_shrink) ** 2
            )
        )
        x_node_rot, y_node_rot, z_node_rot = proper_rotation(
            slice_angle, upright_angle, x_node, y_node, z_node, shrink_factor
        )
        rot_eucl_distance.append(
            sqrt(
                (x_node_rot - x_soma_rot) ** 2
                + (y_node_rot - y_soma_rot) ** 2
                + (z_node_rot - z_soma_rot) ** 2
            )
        )
        xy_distance.append(sqrt((x_node - x_soma) ** 2 + (y_node - y_soma) ** 2))
        xy_rot_distance.append(
            sqrt((x_node_rot - x_soma_rot) ** 2 + (y_node_rot - y_soma_rot) ** 2)
        )

    max_eucl_distance = max(eucl_distance)
    max_shrinked_eucl_distance = max(shrinked_eucl_distance)
    max_rot_eucl_distance = max(rot_eucl_distance)
    max_xy_distance = max(xy_distance)
    max_xy_rot_distance = max(xy_rot_distance)
    # idx_position_far_node = xy_rot_distance.index(max_xy_rot_distance)

    return (
        max_eucl_distance,
        max_rot_eucl_distance,
        max_shrinked_eucl_distance,
        max_xy_distance,
        max_xy_rot_distance,
        # idx_position_far_node,
    )


def calc_distance_from_pia(cell_id, idx_node, cell_feat_orient_new_df):
    """
    Calculate the distance from the pia to a specific node in a cell's morphology.

    This function calculates the distance from the pia (the upper boundary) to a specific
    node of a cell with the given ID (`cell_id`) and node index (`idx_node`). The calculation
    takes into account the cell's orientation-related data, including rotation and shrinkage.

    Parameters:
    -----------
    cell_id : int
        The unique identifier of the cell (specimen_id).

    idx_node : int
        The index of the node for which to calculate the distance from the pia (talken from the morphology swc).

    cell_feat_orient_new_df : pandas.DataFrame
        A DataFrame containing cell features including orientation-related data (I only have angle data fro mice cells).

    Returns:
    --------
    distance_from_pia : float
        The distance from the pia to the specified node.

    Notes:
    ------
    This function calculates the vertical distance from the pia to a specific node in a cell's
    morphology. It accounts for the cell's orientation, including rotation and shrinkage.
    The distance is computed based on the soma's distance from the pia and the node's position
    after applying proper rotation and shrinkage.

    Example:
    --------

    # Specify the cell ID and node index
    >>> cell_id = 479013100

    >>> idx_node = 7

    # Calculate the distance from the pia to the specified node
    >>> distance = calc_distance_from_pia(cell_id, idx_node, cell_feat_orient_new_df)

    >>> print("Distance from Pia:", distance)
    Distance from Pia: 0    353.452268
    dtype: float64

    """

    morph = reconstruct(cell_id)
    morph_df = pd.DataFrame(morph.compartment_list)
    cell_idx = cell_feat_orient_new_df[
        cell_feat_orient_new_df["specimen_id"] == cell_id
    ].index
    soma_distance_from_pia = cell_feat_orient_new_df.loc[
        cell_idx, "soma_distance_from_pia"
    ]
    slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
    upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
    shrink_factor = cell_feat_orient_new_df.loc[cell_idx, "estimated_shrinkage_factor"]
    x_soma, y_soma, z_soma = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df.loc[0, "x"],
        morph_df.loc[0, "y"],
        morph_df.loc[0, "z"],
        shrink_factor,
    )
    x_node, y_node, z_node = proper_rotation(
        slice_angle,
        upright_angle,
        morph_df.loc[idx_node, "x"],
        morph_df.loc[idx_node, "y"],
        morph_df.loc[idx_node, "z"],
        shrink_factor,
    )
    y_pia = y_soma + soma_distance_from_pia
    distance_from_pia = y_pia - y_node
    return distance_from_pia




def apply_filters(spex, area, layer, neur_type, cell_feat_orient_df):
    """
    Filter and return a DataFrame containing cells based on species, brain area, layers, and spiny/aspiny features.

    This function filters a DataFrame of cell features based on multiple conditions, including species ('spex'),
    brain area ('area'), layers ('layer'), and dendrite type ('neur_type'). It returns a DataFrame containing cells
    that satisfy all specified conditions.

    Parameters:
    -----------
    spex : str
        The species condition. Options are 'Homo Sapiens' for humans or 'Mus Musculus' for mice.

    area : str
        The brain area condition. Options depend on the species:
        - For humans: 'MTG', 'MFG', 'AnG', 'PLP', 'TemL', 'ITG', 'IFG', 'FroL', 'SFG'.
        - For mice: 'VISp', 'VISpor', 'VISpm', 'VISal', 'VISl', 'VISrl', 'VISam', 'VISli', 'VISpl'.

    layer : str
        The layer condition. Options are '1', '2', '2/3', '3', '4', '5', '6', '6a', '6b'
        ('2/3','6a', '6b' are only for mice) .

    neur_type : str
        The dendrite type condition. Options are 'spiny' or 'aspiny'.

    cell_feat_orient_df : pandas.DataFrame
        A DataFrame containing cell features, including species, brain area, layers, and dendrite type information.

    Returns:
    --------
    filtered_df : pandas.DataFrame
        A DataFrame containing cells that meet all specified conditions.

    Notes:
    ------
    This function filters cells based on multiple conditions, including species, brain area, layers, and dendrite type.
    It returns a DataFrame containing the cells that satisfy all specified conditions.

    Example:
    --------

    # Apply multiple filters
    >>> spex = 'Homo Sapiens'
    >>> area = 'MTG'
    >>> layer = '2'
    >>> neur_type = 'spiny'
    >>> filtered_cells = apply_filters(spex, area, layer, neur_type, cell_feat_orient_df)

    >>> print("Filtered cells:")
    >>> print(filtered_cells)
    Filtered cells:
            Unnamed: 0  adaptation     avg_isi  electrode_0_pa  f_i_curve_slope  ...  me-type  upright_angle  soma_distance_from_pia  estimated_shrinkage_factor  estimated_slice_angle
    5             5    0.306257  168.450000       -7.342500         0.072630  ...      NaN            NaN                     NaN                         NaN                    NaN
    22           22    0.428605  211.626667      -19.685000         0.066221  ...      NaN            NaN                     NaN                         NaN                    NaN
    36           36    0.030953   92.260000       -0.560000         0.126314  ...      NaN            NaN                     NaN                         NaN                    NaN
    159         159    0.028581   72.390769       -8.322500         0.205730  ...      NaN            NaN                     NaN                         NaN                    NaN
    203         203    0.072791  137.666667      -41.535001         0.107833  ...      NaN            NaN                     NaN                         NaN                    NaN
    215         215    0.279993  208.750000       -9.100000         0.083853  ...      NaN            NaN                     NaN                         NaN                    NaN
    323         323         NaN  248.360000       18.430001         0.071813  ...      NaN            NaN                     NaN                         NaN                    NaN
    370         370    0.017420   78.634545      -14.050000         0.154966  ...      NaN            NaN                     NaN                         NaN                    NaN
    391         391    0.053403  130.500000      -80.159990         0.141667  ...      NaN            NaN                     NaN                         NaN                    NaN
    468         468         NaN         NaN      -11.765000         0.018452  ...      NaN            NaN                     NaN                         NaN                    NaN
    539         539    0.072607  228.540000      -29.330000         0.065784  ...      NaN            NaN                     NaN                         NaN                    NaN

    """

    species_cells = cell_feat_orient_df[
        cell_feat_orient_df["donor__species"].values == spex
    ].index
    area_cells = cell_feat_orient_df[
        cell_feat_orient_df["structure_parent__acronym"].values == area
    ].index
    layer_cells = cell_feat_orient_df[
        cell_feat_orient_df["structure__layer"].values == layer
    ].index
    neur_type_cells = cell_feat_orient_df[
        cell_feat_orient_df["tag__dendrite_type"].values == neur_type
    ].index
    species_area_layer_neur_type_cells = (
        set(species_cells) & set(area_cells) & set(layer_cells) & set(neur_type_cells)
    )
    species_area_layer_neur_type_idx = np.array(
        list(species_area_layer_neur_type_cells)
    )
    species_area_layer_neur_type_idx.sort()
    species_area_layer_neur_type_df = cell_feat_orient_df.loc[
        species_area_layer_neur_type_idx
    ]
    return species_area_layer_neur_type_df


##########################################################################################################
# VARIABLE PARAMETERS 
# Variable parameters to try out the functions and have a general example of how they work.
# I got these values from the cell_feat_orient_df, looking at the columns that I was 
# interested in (e.g. 'structure__layer', 'donor__species', 'tag__dendrite_type', 'estimated_slice_angle',
# 'upright_angle', 'estimated_shrinkage_factor', etc.)

download_recos = False

cell_id = 479013100

layer = "2/3"  # '1', '2', '3', '4', '5', '6', '6a', '6b'

spex = "Mus musculus"  # 'Homo Sapiens'

neur_type = "spiny"  # 'aspiny'

slice_angle = 0  

upright_angle = 176.909187120959  

shrink_factor = 3.05757172357999  

##########################################################################################################
# COMPUTATION W VARIABLE PARAMETERS 

# If I need to rebuild or modify the cell_feat_df or the other DataFrames, I can uncomment these lines.
# ef_df, mor_df, feat_df = get_features_df()
# cell_feat_df = deepcopy(feat_df)
# needed_columns = [
#     "specimen__id",
#     "specimen__name",
#     "specimen__hemisphere",
#     "structure__id",
#     "structure__name",
#     "structure__acronym",
#     "structure_parent__id",
#     "structure_parent__acronym",
#     "structure__layer",
#     "nr__average_parent_daughter_ratio",
#     "nrwkf__id",
#     "erwkf__id",
#     "ef__avg_firing_rate",
#     "si__height",
#     "si__width",
#     "si__path",
#     "csl__x",
#     "csl__y",
#     "csl__z",
#     "csl__normalized_depth",
#     "cell_reporter_status",
#     "m__glif",
#     "m__biophys",
#     "m__biophys_perisomatic",
#     "m__biophys_all_active",
#     "tag__apical",
#     "tag__dendrite_type",
#     "morph_thumb_path",
#     "ephys_thumb_path",
#     "ephys_inst_thresh_thumb_path",
#     "donor__age",
#     "donor__sex",
#     "donor__disease_state",
#     "donor__race",
#     "donor__years_of_seizure_history",
#     "donor__species",
#     "donor__id",
#     "donor__name",
# ]
# not_needed_columns = [
#     "nr__number_bifurcations",
#     "nr__average_contraction",
#     "nr__reconstruction_type",
#     "nr__max_euclidean_distance",
#     "nr__number_stems",
#     "ef__fast_trough_v_long_square",
#     "ef__upstroke_downstroke_ratio_long_square",
#     "ef__adaptation",
#     "ef__f_i_curve_slope",
#     "ef__threshold_i_long_square",
#     "ef__tau",
#     "ef__avg_isi",
#     "ef__ri",
#     "ef__peak_t_ramp",
#     "ef__vrest",
#     "line_name",
# ]
# cell_feat_df = cell_feat_df.reindex(
#     columns=cell_feat_df.columns.tolist() + needed_columns
# )

# the dataframe cell_feat_df is the one that contains all the features of the cells, but it's not complete yet because it has no info on angle, shrinkage, etc.
cell_feat_df = pd.read_csv(f"{str(data_root_path)}/cell_feat_data.csv")


# # To see if a cell has morphological reconstruction
# if download_recos == True:
#     for this_row in feat_df.index:
#         try:
#             file_name = f'specimen_id_{feat_df.loc[this_row,"specimen_id"]}'
#             full_filename = output_dir / file_name
#             print(full_filename)
#             ex = ctc.get_reconstruction(
#                 feat_df.loc[this_row, "specimen_id"], file_name=full_filename
#             )
#         except Exception:
#             print(
#                 f'Reco not found for cell {feat_df.loc[this_row,"specimen_id"]} at row={this_row}'
#             )

# # I am merging the data from this two dataframes (cell_feat_df and cell_df) in order to have these info in one dataframe (cell_feat_df)
# cell_df = pd.read_csv(f"{str(data_root_path)}/cell_types_specimen_details_3.csv")

# common_id = set(cell_df.specimen__id) & set(feat_df.specimen_id)

# for cell_id in common_id:
#     id_row = cell_df.loc[cell_df["specimen__id"] == cell_id, needed_columns]
#     specimen__id = id_row["specimen__id"].values
#     row_index = cell_feat_df[cell_feat_df["specimen_id"].values == specimen__id].index
#     cell_feat_df.loc[row_index, needed_columns] = id_row.values


# # I am merging the data from this two dataframes (cell_feat_df and cell_feat_orient_df) in order to have all the info (also on oritntation)
# # in one dataframe (cell_feat_df); I commented this part because I have already done it and I saved the new dataframe in a csv file.

# orientation_df = pd.read_csv(
#     f"{str(data_root_path)}/orientation_data.csv"
# )  # This dataframe contains only the data on agles, shrinkage, etc. but only for mice cells 
# orient_id = set(orientation_df.specimen_id) & set(cell_feat_df.specimen_id)
# cell_feat_orient_df = deepcopy(cell_feat_df)
# cell_feat_orient_df[list(orientation_df.columns)] = pd.DataFrame(
#     [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
#     index=cell_feat_orient_df.index,
# )
# for cell_id in orient_id:
#     id_row = orientation_df.loc[
#         orientation_df["specimen_id"] == cell_id, list(orientation_df.columns)
#     ]
#     specimen__id = id_row["specimen_id"].values
#     row_index = cell_feat_orient_df[cell_feat_df["specimen_id"].values == cell_id].index
#     cell_feat_orient_df.loc[row_index, list(orientation_df.columns)] = id_row.values


# This dataframe is the definitive one, cointaining all the data for all the cells, including angles, shrinkage, etc.;
# it has nan values for the cells that are not mice on angle columns.
cell_feat_orient_df = pd.read_csv(f"{str(data_root_path)}/cell_feat_orientation_data.csv")


# I commented these lines because they are not necessary, but if one don't want the nan values in the angle columns
# for the human cells, they can be uncommented.

# cell_feat_orient_new_df = cell_feat_orient_df.dropna(
#     subset=["soma_distance_from_pia"]
# )
# It's the dataframe that contains only data of cells whose angles, shrinkage, etc.,  is not nan
# cell_feat_orient_new_df = pd.read_csv(f"{str(data_root_path)}/oriented_data.csv")


# Here I analyzed the morphology of the mice cells (because are the only one that have angle data), 
# making a list of their specimens. 
# I removed from those specimens the ones that were repeated (because the DataFrame contains two rows for the 
# same cell) and that caused several problems in the analysis.
# I have divided the left ones in 3 lists:
# the list 'good_specimens' contains the ids of the cells that are ok (I have checked them by looking at the hist
# of the soma_distance_from_pia and the soma_distance_from_pia_shrinked); the list 'not_problematic_specimens'
# contains the ids of the cells that seemed to have problems (like looking at the hist using the method
# soma_coord_and_pia_distance), but after the rotation and shrinkage correction they were ok; the list
# 'problematic_specimens' contains the ids of the cells that I had to remove from the list because their morphology
# was wrong also after the rotation and shrinkage correction (like there were some branches that were going over the
# pia layer); I commented this last list, but I left it here because it can be useful to see which cells were
# problematic.


good_specimens = [
    479013100,
    582644266,
    501799874,
    497611660,
    535708196,
    586073683,
    478888052,
    487664663,
    554807924,
    478499902,
    478586295,
    510715606,
    569723367,
    485837504,
    479010903,
    471141261,
    314900022,
    512322162,
    585832440,
    502999078,
    573404307,
    476049169,
    480351780,
    580162374,
    386049446,
    397353539,
    488117124,
    574067499,
    485184849,
    567354967,
    591268268,
    485835016,
    589760138,
    480114344,
    530737765,
    515524026,
    583146274,
    562541627,
    574734127,
    476616076,
    333785962,
    476048909,
    471087830,
    585952606,
    476451456,
    471767045,
    480003970,
    480116737,
    483020137,
    515986607,
    594091004,
    321906005,
    565863515,
    569723405,
    609435731,
    515249852,
    422738880,
    487601493,
    471786879,
    580010290,
    480124551,
    555345752,
    476126528,
    478892782,
    584231995,
    557037024,
    486111903,
    488501071,
    475202388,
    580161872,
    475068599,
    519749342,
    510658021,
    485835055,
    586071425,
    561517025,
    476131588,
    471077857,
    584872371,
    584680861,
    585944237,
    469798159,
    502359001,
    484679812,
    486110216,
    563226105,
    479770916,
    313862373,
    354190013,
    324025371,
    485931158,
    536951541,
    485912047,
    323865917,
    570896413,
    571311039,
    526531616,
    560678143,
    341442651,
    475744706,
    468193142,
    561985849,
    577369606,
    502269786,
    483061182,
    602822298,
    567007144,
    313862022,
    554779051,
    607124114,
    565855793,
    487661754,
    490205998,
    483068687,
    563180078,
    574993444,
    515435668,
    517319635,
    565880475,
    561940338,
    560753350,
    502978383,
    561325425,
    480169178,
    574036994,
    578485753,
    382982932,
    473020156,
    488687894,
    586379590,
    592952680,
    314642645,
    527826878,
    510136749,
    486146717,
    512319604,
    562003142,
    567312865,
    517077548,
    466245544,
    479728896,
    571379222,
    585925172,
    485574832,
    473601979,
    554454458,
    395830185,
    486893033,
    530731648,
    571396942,
    488697163,
    490387590,
    477880128,
    325941643,
    509515969,
    469793303,
    575642695,
    490259231,
    584254833,
    598628992,
    485909730,
    488698341,
    479905853,
    589442285,
    476054887,
    571306690,
    476455864,
    589427435,
    483101699,
    585830272,
    488680917,
    509003464,
    578774163,
    580005568,
    486262299,
    318733871,
    515464483,
    354833767,
    534303031,
    583138568,
    514767977,
    479225052,
    324493977,
    560965993,
    586072188,
    485161419,
    490916882,
    599334696,
    555241875,
    565888394,
    476562817,
    488504814,
    571867358,
    329550277,
    348592897,
    579626068,
    480087928,
    583104750,
    614777438,
    565462089,
    473943881,
    539014038,
    586072464,
    469992918,
    468120757,
    469704261,
    564349611,
    479179020,
    557874460,
    488695444,
    555241040,
    555019563,
    586566174,
    589128331,
    569809287,
    580145037,
    570896453,
    320668879,
    561532710,
    397351623,
    526668864,
    566647353,
    323452196,
    490485142,
    569998790,
    473543792,
    473564515,
    485468180,
    324025297,
    564346637,
    565209132,
    475459689,
    474626527,
    566761793,
    514824979,
    565120091,
    501845630,
    623185845,
    570438732,
    386970660,
    522301604,
    502366405,
    603402458,
    566678900,
    590558808,
    565407476,
    486025194,
    526643573,
    471143169,
    323452245,
    471129934,
    481093525,
    476218657,
    483092310,
    574059157,
    314804042,
    476686112,
    577218602,
    566541912,
    324266189,
    471077468,
    557261437,
    557998843,
    513510227,
    582613001,
    603423462,
    482516713,
    327962063,
    506133241,
    477135941,
    547344442,
    547325858,
    584682764,
    585951863,
    476216637,
    508980706,
    488679042,
    568568071,
    502267531,
    333604946,
    517345160,
    488683425,
    586073850,
    560992553,
    567029686,
    574377552,
    567320213,
    527869035,
    522152249,
    517647182,
    480353286,
    501841672,
    574992320,
    486132712,
    486239338,
    501850666,
    567927838,
    507101551,
    605660220,
    569958754,
    318808427,
    521938313,
    467703703,
    471800189,
    481136138,
    503814817,
    526950199,
    485911892,
    475057898,
    320207387,
    480122859,
    588712191,
    473611755,
    471789504,
    580007431,
    476135066,
    515315072,
    513800575,
    507918877,
    513531471,
    557864274,
    585946742,
    583434059,
    575774870,
    559387643,
    509881736,
    565415071,
    588402092,
    555697303,
    484564503,
    490382353,
    502367941,
    475623964,
    503286448,
    557252022,
    485838981,
    479225080,
    501847931,
    578938153,
    502383543,
    502614426,
    471758398,
    566671538,
    349621141,
    569670455,
    476263004,
    591278744,
    476457450,
    488420599,
    527116037,
    591265629,
    324521027,
    569965244,
    580014328,
    516362762,
    486502127,
    585805211,
    593321019,
    478497530,
    503823047,
    586559181,
    586072783,
    526573598,
    485880739,
    486052980,
    524850271,
    547262585,
    501956013,
    587045566,
    370351753,
    565459685,
]

not_problematic_specimens = [
    567952169,
    501845152,
    329550137,
    486560376,
    524689239,
    579662957,
    582917630,
    515771244,
    585841870,
    488680211,
    396608557,
    476823462,
    479704527,
    466632464,
    555001065,
    574038330,
    478058328,
    535728342,
    573622646,
    570946690,
    471410185,
    485836906,
    572609108,
    314831019,
    583138230,
    314822529,
    485574721,
    614767057,
]

# problematic_specimens = [475585413, 521409057, 527095729, 487099387]

################################################################################################################################
# VISUALIZATION

# As an example, I called most of the methods for visualization for the cell with id 479013100, but they can be called for any cell.
# I commented them because I don't want to run them every time I run the script, but they can be uncommented to try them out.

viz = Viz()
cell_id = 479013100
specimens = [
    479013100,
    582644266,
    501799874]
# This is the dataframe that contains only data of cells whose angles, shrinkage, etc.,  is not nan
# (I only have these data for mice cells)
cell_feat_orient_new_df = pd.read_csv(f"{str(data_root_path)}/oriented_data.csv")
# This is the list of the indexes of the mice cells in the dataframe cell_feat_orient_new_df that are in the VISp area
VISp_mice_cells_idx = [cell_feat_orient_new_df.index[cell_feat_orient_new_df["structure_parent__acronym"].values == "VISp"]]
VISp_mice_cells_idx_list = VISp_mice_cells_idx[0].tolist()

# This is to visualize the neuron's 2D shape in the xy plane using morphology coordinates
# viz.show_neuron_2D(cell_id)

# This is to visualize the neuron's 2D shape in the xy plane after proper rotation
# viz.show_orient_neuron_2D(cell_id, cell_feat_orient_new_df)

# This is to visualize the neuron's 3D shape using properly rotated coordinates
# viz.show_orient_neuron_3D(cell_id, cell_feat_orient_new_df)

# This is to show the difference between maximum and minimum y coordinates (rotated) of neurons, compared to PCA coordinates
# viz.y_coord_difference(specimens, cell_feat_orient_new_df)

# This plots histograms of soma distance from pia and soma y-coordinate (after rotation)
# viz.soma_coord_and_pia_distance(specimens, cell_feat_orient_new_df)

# This plots a single neuron (2D shape) with proper rotation and origin at the pia
# viz.plot_morphology_from_pia(cell_id, cell_feat_orient_new_df)

# This plots a histogram of the cortical depth of neurites for a given neuron
# viz.cortical_depth_hist(cell_id, cell_feat_orient_new_df)

# Plots histograms of soma distances from the pia for cells in the index list of VISp mice cells;
# the method returns the data of the histograms, which can be used to plot the layer boundaries (calling the method layer_boundaries)
# data1, data2, data2_3, data3, data4, data5, data6, data6a, data6b = viz._soma_distance_hist_layer(VISp_mice_cells_idx_list, cell_feat_orient_new_df)

# This plots histogram of layer distribution and boundaries between layers for VISp mice cells 
# the method returns also the intersection points between the histograms (so the boundaries between layers)
# It can be run only after getting the data from the method _soma_distance_hist_layer
# data1, data2, data2_3, data3, data4, data5, data6, data6a, data6b = viz._soma_distance_hist_layer(VISp_mice_cells_idx_list, cell_feat_orient_new_df)
# viz.layer_boundaries(data1, data2_3, data4, data5, data6a, data6b)

# This is the plot of the 2D morphology of the cell, within its corresponding layer 
viz.plot_in_layer(cell_id, cell_feat_orient_new_df, VISp_mice_cells_idx_list)

plt.show()
