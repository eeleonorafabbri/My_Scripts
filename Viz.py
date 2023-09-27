"""
Viz

Description:
    This script contains methods for visualizing the morphology of neurons from the Allen Institute's
    Cell Types Database. It is linked to the Organised_Script.py script, which contains methods for
    organizing the data from the Cell Types Database.

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
import csv
from sklearn.decomposition import PCA  
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve, newton, bisect

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import ReporterStatus as RS
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.api.queries.cell_types_api import CellTypesApi

# from examples.rotation_cell_morph_example import morph_func as mf
# import Organised_Script


from pathlib import Path
import math
import pprint
import pdb


# Most of the methods in this class are taken from the Allen Institute's example script (ateam-tools/examples/rotation_cell_morph_example/morph_func.py)
class Analysis:
    def __init__(self, output_dir, ctc):
        self.output_dir = Path("/opt3/Eleonora/data/reconstruction")
        self.ctc = CellTypesCache(manifest_file=self.output_dir / "manifest.json")

    def reconstruct(self, id_cell):
        '''
        Returns the morphology (the swc file) of the cell with its specimen_id, in which 
        there are id, type, the coordinates of each node of the cell, r and parent_id 
        '''
        morphology = self.ctc.get_reconstruction(id_cell)
        return morphology

    def proper_rotation(self, slice_angle, upright_angle, x1, y1, z1, shrink_factor):
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


# ateam-tools method
    def get_cell_morphXYZ(self, cell_id):
        """
        Obtain the morphology location data for PCA analysis.

        This method retrieves the 3D coordinates (X, Y, Z) of a cell's morphology components
        relative to the soma for Principal Component Analysis (PCA). It also provides the soma's
        coordinates as the reference point (0, 0, 0).

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the cell (= specimen_id).

        Returns:
        --------
        morph_data : numpy.ndarray
            A NumPy array containing the relative coordinates (X, Y, Z) of the morphology components
            (axon, dendrite, soma) with respect to the soma.

        morph_soma : list
            A list containing the absolute coordinates (X, Y, Z) of the soma.

        Notes:
        ------
        This method retrieves the 3D coordinates of the cell's morphology components (axon, dendrite, soma)
        relative to the soma. The relative coordinates are used for PCA analysis, and the absolute coordinates
        of the soma are provided as a reference point (0, 0, 0).

        Example:
        --------
        >>> viz = Viz()  # Create an instance of the Viz class
        >>> cell_id = 529878215  # Specify the cell ID

        # Obtain morphology data for PCA
        >>> morph_data, morph_soma = viz.get_cell_morphXYZ(cell_id)

        >>> print("Morphology Data (Relative to Soma):")
        >>> print(morph_data)
        Morphology Data (Relative to Soma):
        [[  0.       0.       0.    ]
        [ -1.6908  -5.7886   0.4631]
        [ -1.7011  -8.0686   0.6611]
        ...
        [114.3943 144.4255 -18.8484]
        [114.956  145.4207 -18.9434]
        [116.0279 145.812  -19.04  ]]

        >>> print("Soma Coordinates (Absolute):")
        >>> print(morph_soma)
        Soma Coordinates (Absolute):
        [509.5971, 638.4961, 32.76]
        
        """

        morph = self.ctc.get_reconstruction(cell_id)
        x = []
        y = []
        z = []
        for n in morph.compartment_list:
            # print(n['type']) #type=1, soma; type=2, axon; type=3, dendrite; type=4,apical dendrite
            if n["type"] == 4 or n["type"] == 3 or n["type"] == 1:
                x.append(n["x"] - morph.soma["x"])
                y.append(n["y"] - morph.soma["y"])
                z.append(n["z"] - morph.soma["z"])

        morph_data = np.array(np.column_stack((x, y, z)))
        morph_soma = [morph.soma["x"], morph.soma["y"], morph.soma["z"]]

        return (morph_data, morph_soma)
    

 # ateam-tools method
    def cal_rotation_angle(self, morph_data):
        """
        Calculate rotation angles to align morphology with a reference direction (Y-axis).

        This method performs a series of rotations on a morphology dataset to align it with a reference
        direction (typically the Y-axis) while preserving its shape. It calculates the rotation angles
        required to achieve this alignment.

        Parameters:
        -----------
        morph_data : numpy.ndarray
            A NumPy array containing the morphology data to be rotated.

        Returns:
        --------
        v1 : numpy.ndarray
            The original orientation vector before any rotations.

        theta : list
            A list of three rotation angles (in radians) [anglex, 0, anglez] to achieve alignment
            with the reference direction.

        Notes:
        ------
        This method applies rotations to align the morphology with a reference direction. The
        calculated angles (anglex and anglez) are used to rotate the morphology while preserving its shape.
        """

        pca = PCA(n_components=2)
        pca.fit(morph_data)
        proj = morph_data.dot(
            pca.components_[0]
        )  # the projection of morphology on the direction of first pca
        # v1 = -1*pca.components_[0]  # the first principal component, when apical dendrite goes down
        # v1 = 1*pca.components_[0]  # the first principal component
        v1 = np.sign(proj.mean()) * pca.components_[0]
        # The goal is to rotate v1 to parallel to y axis
        x1 = v1[0]
        y1 = v1[1]
        z1 = v1[2]
        # First rotate in the anticlockwise direction around z axis untill x=0
        v2 = [0, math.sqrt(y1 * y1 + x1 * x1), z1]
        dv = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
        anglez = 2 * math.asin(
            math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) * 0.5 / v2[1]
        )
        if x1 < 0:  # when x1 in the negative side, change the sign
            anglez = -anglez
        # Second rotate in the anticlockwise direction round x axis untill z = 0
        v3 = [0, math.sqrt(x1 * x1 + y1 * y1 + z1 * z1), 0]
        dv2 = [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]]
        anglex = -2 * math.asin(
            math.sqrt(dv2[0] * dv2[0] + dv2[1] * dv2[1] + dv2[2] * dv2[2]) * 0.5 / v3[1]
        )
        if z1 < 0:  # when z1 in the negative side, change the sign
            anglex = -anglex
        theta = [anglex, 0, anglez]
        R = self.eulerAnglesToRotationMatrix(theta)
        v3_hat = R.dot(v1)

        return (v1, theta)
    

# ateam-tools method
    def eulerAnglesToRotationMatrix(self, theta):
        """
        Calculate a rotation matrix from Euler angles.

        This method computes a 3x3 rotation matrix from a given set of Euler angles (yaw, pitch, and roll).
        The rotation matrix describes the transformation of a 3D coordinate system based on the Euler angles.

        Parameters:
        -----------
        theta : list
            A list of three Euler angles in radians, representing yaw, pitch, and roll, respectively.

        Returns:
        --------
        R : numpy.ndarray
            A 3x3 NumPy array representing the rotation matrix.

        Notes:
        ------
        This method calculates a rotation matrix based on the provided Euler angles, which describe how a
        coordinate system is rotated in 3D space. The resulting rotation matrix can be used to transform
        3D points or vectors.
        """

        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(theta[0]), -math.sin(theta[0])],
                [0, math.sin(theta[0]), math.cos(theta[0])],
            ]
        )

        R_y = np.array(
            [
                [math.cos(theta[1]), 0, math.sin(theta[1])],
                [0, 1, 0],
                [-math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )

        R_z = np.array(
            [
                [math.cos(theta[2]), -math.sin(theta[2]), 0],
                [math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(R_x, np.dot(R_y, R_z))

        return R


# ateam-tools method
    def cell_morphology_rot(self, cell_id, x_soma, y_soma, z_soma, theta):
        """
        Rotate and translate a cell's morphology for visualization.

        This method applies a series of 3D rotations and translations to a cell's morphology to align
        it with a specified orientation and translate it to a new soma location for visualization.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the cell (= specimen_id).

        x_soma : float
            The new X-coordinate of the soma's location after translation.

        y_soma : float
            The new Y-coordinate of the soma's location after translation.

        z_soma : float
            The new Z-coordinate of the soma's location after translation.

        theta : list
            A list of three Euler angles [anglex, angley, anglez] in radians to specify the rotation
            angles around the X, Y, and Z axes, respectively.

        Returns:
        --------
        morph : object
            The rotated and translated morphology object for visualization.

        Notes:
        ------
        This method applies a series of 3D rotations and translations to the cell's morphology.
        It first rotates the morphology based on the provided Euler angles (theta) and then translates
        the soma location to the specified coordinates (x_soma, y_soma, z_soma).

        """

        theta_z = theta[2]
        theta_y = theta[1]
        theta_x = theta[0]
        morph = self.ctc.get_reconstruction(cell_id)
        # First applying a rotation angle around z axis
        tr_rot_z = [
            math.cos(theta_z),
            -math.sin(theta_z),
            0,
            math.sin(theta_z),
            math.cos(theta_z),
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
        # Second applying a rotation angle around y axis
        tr_rot_y = [
            math.cos(theta_y),
            0,
            math.sin(theta_y),
            0,
            1,
            0,
            -math.sin(theta_y),
            0,
            math.cos(theta_y),
            0,
            0,
            0,
        ]
        # Third applying a rotation angle around x axis
        tr_rot_x = [
            1,
            0,
            0,
            0,
            math.cos(theta_x),
            -math.sin(theta_x),
            0,
            math.sin(theta_x),
            math.cos(theta_x),
            0,
            0,
            0,
        ]

        morph.apply_affine(tr_rot_z)
        morph.apply_affine(tr_rot_y)
        morph.apply_affine(tr_rot_x)
        # translate the soma location
        tr_soma = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            -morph.soma["x"] + x_soma,
            -morph.soma["y"] + y_soma,
            -morph.soma["z"] + z_soma,
        ]
        morph.apply_affine(tr_soma)
        return morph


# ateam-tools method
    def plot_cell_morph_xyzy(self, axes, morph):
        soma_col = [134.0 / 255.0, 134.0 / 255.0, 148.0 / 255.0]
        axon_col = [93.0 / 255.0, 127.0 / 255.0, 177.0 / 255.0]
        dend_col = [153.0 / 255.0, 40.0 / 255.0, 39.0 / 255.0]
        apical_dend_col = [227.0 / 255.0, 126.0 / 255.0, 39.0 / 255.0]
        ap = 1

        for n in morph.compartment_list:
            for c in morph.children_of(n):
                if n["type"] == 2:
                    axes[0].plot(
                        [n["x"], c["x"]], [n["y"], c["y"]], color=axon_col, alpha=ap
                    )
                    axes[1].plot(
                        [n["z"], c["z"]], [n["y"], c["y"]], color=axon_col, alpha=ap
                    )
                if n["type"] == 3:
                    axes[0].plot(
                        [n["x"], c["x"]], [n["y"], c["y"]], color=dend_col, alpha=ap
                    )
                    axes[1].plot(
                        [n["z"], c["z"]], [n["y"], c["y"]], color=dend_col, alpha=ap
                    )
                if n["type"] == 4:
                    axes[0].plot(
                        [n["x"], c["x"]],
                        [n["y"], c["y"]],
                        color=apical_dend_col,
                        alpha=ap,
                    )
                    axes[1].plot(
                        [n["z"], c["z"]],
                        [n["y"], c["y"]],
                        color=apical_dend_col,
                        alpha=ap,
                    )
                if n["type"] == 1:  # soma
                    axes[0].scatter(
                        n["x"], n["y"], s=math.pi * (n["radius"] ** 2), color=soma_col
                    )
                    axes[1].scatter(
                        n["z"], n["y"], s=math.pi * (n["radius"] ** 2), color=soma_col
                    )

        axes[0].set_ylabel("y")
        axes[0].set_xlabel("x")
        axes[1].set_xlabel("z")
        self.simpleaxis(axes[0])
        self.simpleaxis(axes[1])


# ateam-tools method
    def simpleaxis(self, ax):
        """
        Simplify the axis by hiding the right and top spines.

        This method simplifies the appearance of a Matplotlib axis (ax) by hiding the right and top
        spines and only showing ticks on the left and bottom spines.

        Parameters:
        -----------
        ax : matplotlib.axes._axes.Axes
            The Matplotlib axis to be modified.

        Notes:
        ------
        This method is useful for creating clean and minimalistic plots by removing unnecessary
        spines from the axis.
        """

        # Hide the right and top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


# ateam-tools method
    def get_rotation_theta(self, cell_id):
        """
        Get the rotation angles for a specific cell.

        This method calculates the rotation angles (yaw, pitch, and roll) required to align a specific cell's
        morphology with a reference orientation.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the cell (= specimen_id).

        Returns:
        --------
        theta : list
            A list of three Euler angles [yaw, pitch, roll] in radians, representing the rotation angles
            around the Y, X, and Z axes, respectively.

        Notes:
        ------
        This method retrieves the morphology data for the specified cell, calculates the rotation angles
        required to align the morphology with a reference orientation, and returns the Euler angles.
        """

        # get the morphology data (from apical dendrite, dendrite, and soma) used for PCA
        [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)
        [v, theta] = self.cal_rotation_angle(morph_data)
        return theta


class Viz(Analysis):
    def __init__(self):
        self.axon_color = "blue"
        self.bas_dendrite_color = "red"
        self.api_dendrite_color = "orange"
        self.output_dir = Path("/opt3/Eleonora/data/reconstruction")
        self.ctc = CellTypesCache(manifest_file=self.output_dir / "manifest.json")
        # self.ct = CellTypesApi()

    def show_neuron_2D(self, cell_id):
        """
        Visualize the neuron's 2D shape in the xy plane using morphology coordinates.

        This method plots and visualizes the shape of a neuron in the 2D (xy) plane using the
        coordinates provided in the morphology file. Different compartments (axon, basal dendrite,
        and apical dendrite) are plotted in different colors for easy differentiation.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the neuron cell (= specimen_id).

        Notes:
        ------
        This method retrieves the morphology data for the specified cell, separates it into different
        compartments (axon, basal dendrite, and apical dendrite), and visualizes their shapes in the xy plane.
        """

        # Get the morphology file using the cell_id of the neuron
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            ax.scatter(df["x"], df["y"], color=color)
        ax.invert_yaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def show_orient_neuron_2D(self, cell_id, cell_feat_orient_new_df):
        """
        Visualize the neuron's 2D shape in the xy plane after proper rotation.

        This method plots and visualizes the shape of a neuron in the 2D (xy) plane after applying
        proper rotation (function in Organised_Script.py) using the slice_angle, upright_angle, and shrinkage factors from the
        `cell_feat_orient_new_df` dataframe.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the neuron cell (= specimen_id).
        cell_feat_orient_new_df : pandas.DataFrame
            A dataframe containing orientation-related features for neurons.

        Notes:
        ------
        This method retrieves the morphology data for the specified cell, applies proper rotation
        using the orientation parameters from `cell_feat_orient_new_df`, and visualizes the rotated
        shapes of different compartments (axon, basal dendrite, and apical dendrite) in the xy plane.
        """

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            ax.scatter(x_coord, y_coord, color=color)
        ax.invert_yaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def show_orient_neuron_3D(self, cell_id, cell_feat_orient_new_df):
        """
        Visualize the neuron's 3D shape using properly rotated coordinates.

        This method plots and visualizes the 3D shape of a neuron using the coordinates that
        have been properly rotated based on the slice_angle, upright_angle, and shrinkage factors
        provided in the `cell_feat_orient_new_df` dataframe.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the neuron cell (= specimen_id).
        cell_feat_orient_new_df : pandas.DataFrame
            A dataframe containing orientation-related features for neurons.

        Notes:
        ------
        This method retrieves the morphology data for the specified cell, applies proper rotation
        using the orientation parameters from `cell_feat_orient_new_df`, and visualizes the rotated
        shapes of different compartments (axon, basal dendrite, and apical dendrite) in 3D space.
        """

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
        ax.scatter(x_coord, z_coord, y_coord, color=color)
        ax.invert_yaxis()  # Maybe it is not necessary
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def _plot_difference(self, cell_id, cell_feat_orient_new_df):
        """
        Create three different plots: two from PCA coordinates and one from rotated coordinates.

        This method generates three different plots to visualize the difference between the morphology
        before and after rotation. The first two plots are based on PCA coordinates, showing the
        principal vectors before and after rotation. The third plot displays the morphology after proper
        rotation using the slice_angle and upright_angle from `cell_feat_orient_new_df`.

        Parameters:
        -----------
        cell_id : int
            The unique identifier of the neuron cell (= specimen_id).
        cell_feat_orient_new_df : pandas.DataFrame
            A dataframe containing orientation-related features for neurons.

        Notes:
        ------
        This method computes the rotation matrix based on orientation angles, rotates the principal
        vectors before and after rotation, and visualizes the neuron's morphology in different axes.
        """

        # FIRST PLOT: PCA COORDINATES

        theta = list(np.around(self.get_rotation_theta(cell_id), decimals=6))

        # get the morphology data (from apical dendrite, dendrite, and soma) used for PCA
        [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)

        # get the first principal vector and the rotation angle
        [v, theta] = self.cal_rotation_angle(morph_data)

        # Based on the rotation angle to calculate the rotation matrix R
        R = self.eulerAnglesToRotationMatrix(theta)  # rotation matrix

        # the first principal component before and after rotated
        v = v * 400
        v_rot = R.dot(v)
        # The morphology locations used for PCA after rotations
        X_rot = np.array(morph_data)  # The rotated position of new x,y,z
        for i in range(0, len(X_rot)):
            X_rot[i, :] = R.dot(morph_data[i, :])

        # The location of soma, defined by the user
        x_soma = 0
        y_soma = 0
        z_soma = 0
        # The original morphology before rotations
        theta0 = [0, 0, 0]
        morph0 = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta0)

        # Plot the morphology in xy and zy axis
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        self.plot_cell_morph_xyzy(axes, morph0)
        # plot the principal vectors on top of the morphology
        axes[0].plot([x_soma, v[0]], [y_soma, v[1]], color="c")
        axes[1].plot([x_soma, v[2]], [y_soma, v[1]], color="c")
        axes[0].scatter(v[0], v[1], color="blue")
        axes[1].scatter(v[2], v[1], color="blue")

        # The morphology after rotations
        morph_rot = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta)

        # Plot the morphology in xy and zy axis
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        self.plot_cell_morph_xyzy(axes, morph_rot)
        # plot the principal vectors on top of the morphology
        axes[0].plot([x_soma, v_rot[0]], [y_soma, v_rot[1]], color="c")
        axes[1].plot([x_soma, v_rot[2]], [y_soma, v_rot[1]], color="c")
        axes[0].scatter(v_rot[0], v_rot[1], color="blue")
        axes[1].scatter(v_rot[2], v_rot[1], color="blue")

        # SECOND PLOT: ROTATED COORDINATES

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        fig, ax = plt.subplots(1, 1)

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            # shrink_z_coord = df['z'] *shrink_factor
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            ax.scatter(x_coord, y_coord, color=color)
        # ax.invert_xaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])

    def y_coord_difference(self, specimens, cell_feat_orient_new_df):
        """
        Plot a histogram of the difference between maximum and minimum y coordinates (rotated)
        of neurons, compared to PCA coordinates.

        This method calculates the difference between the maximum and minimum y coordinates of neurons
        after proper rotation (using `cell_feat_orient_new_df`), as well as the same difference for PCA
        coordinates. It then plots a histogram to visualize and compare these differences.

        Parameters:
        -----------
        specimens : list
            A list of specimen IDs for the neurons to be analyzed.
        cell_feat_orient_new_df : pandas.DataFrame
            A dataframe containing orientation-related features for neurons.
        """
        
        minmax_difference = []
        for cell_id in specimens:
            cell_idx = cell_feat_orient_new_df[
                cell_feat_orient_new_df["specimen_id"] == cell_id
            ].index
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ].values

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            minmax_difference.append(max(y_coord) - min(y_coord))

        fig, ax = plt.subplots(1, 1)
        plt.hist(
            minmax_difference,
            bins=50,
            orientation="horizontal",
            label="our_rotated_coord",
            color="tab:orange",
        )

        x_soma = 0
        y_soma = 0
        z_soma = 0
        pca_rotated_coord = []
        for cell_id in specimens:
            [morph_data, morph_soma] = self.get_cell_morphXYZ(cell_id)
            [v, theta] = self.cal_rotation_angle(morph_data)
            R = self.eulerAnglesToRotationMatrix(theta)  # rotation matrix
            v = v * 400
            v_rot = R.dot(v)
            X_rot = np.array(morph_data)  # The rotated position of new x,y,z
            for i in range(0, len(X_rot)):
                X_rot[i, :] = R.dot(morph_data[i, :])
            morph_rot = self.cell_morphology_rot(cell_id, x_soma, y_soma, z_soma, theta)
            morph_rot_df = pd.DataFrame(morph_rot.compartment_list)
            pca_rotated_coord.append(max(morph_rot_df["y"]) - min(morph_rot_df["y"]))
        plt.hist(
            pca_rotated_coord,
            bins=50,
            orientation="horizontal",
            label="pca_rotated_coord",
            color="tab:purple",
        )
        plt.legend()

    def soma_coord_and_pia_distance(self, specimens, cell_feat_orient_new_df):
        """
        Plot histograms of soma distance from pia and soma y-coordinate (after rotation).

        This method calculates and plots histograms of two parameters for a list of specimens:
        1. Soma distance from the pia (properly rotated).
        2. Soma y-coordinate (after proper rotation and inversion of the y-axis).

        Parameters:
        -----------
        specimens : list
            A list of specimen IDs for the neurons to be analyzed.
        cell_feat_orient_new_df : pandas.DataFrame
            A dataframe containing orientation-related features for neurons.

        Returns:
        --------
        soma_distance : list
            A list of soma distances from the pia.
        soma_y_coord : list
            A list of soma y-coordinates (after rotation and y-axis inversion).
        """
        
        soma_y_coord = []
        soma_distance = []
        for cell_id in specimens:
            cell_idx = cell_feat_orient_new_df[
                cell_feat_orient_new_df["specimen_id"] == cell_id
            ].index
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ].values

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            soma_y_coord.append(y_coord[0])
            soma_dist = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ].to_list()
            soma_distance.append(soma_dist)
        soma_distance = np.concatenate(soma_distance).tolist()
        soma_y_coord = [-x for x in soma_y_coord]  # because i have inverted the y axis

        plt.hist(
            soma_distance,
            bins=50,
            orientation="horizontal",
            label="soma_distance_from_pia",
            color="yellowgreen",
        )
        plt.hist(
            soma_y_coord,
            bins=50,
            orientation="horizontal",
            label="soma_y_coord",
            color="olivedrab",
        )
        plt.legend()
        return soma_distance, soma_y_coord

    def plot_morphology_from_pia(self, cell_id, cell_feat_orient_new_df):
        """
        Plot a single neuron with proper rotation and origin at the pia.

        This method plots the morphology of a single neuron with proper rotation, where the origin is set at
        the pia (soma_distance_from_pia). The morphology is plotted with different colors for axons, basal dendrites,
        and apical dendrites.

        Parameters:
        -----------
        cell_id : int
            The ID of the neuron cell to be plotted (= specimen_id).
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons.

        Returns:
        --------
        ax : matplotlib.axes._axes.Axes
            The matplotlib axis containing the plotted neuron morphology.
        """

        fig, ax = plt.subplots(1, 1)

        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)

        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        x_soma, y_soma, z_soma = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df.loc[0, "x"],
            morph_df.loc[0, "y"],
            morph_df.loc[0, "z"],
            shrink_factor,
        )
        soma_distance_from_pia = cell_feat_orient_new_df.loc[
            cell_idx, "soma_distance_from_pia"
        ].values
        x_pia = x_soma
        y_pia = y_soma + soma_distance_from_pia

        for d_type, color in [
            [2, self.axon_color],
            [3, self.bas_dendrite_color],
            [4, self.api_dendrite_color],
        ]:
            df = morph_df[morph_df["type"] == d_type]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle, upright_angle, df["x"], df["y"], df["z"], shrink_factor
            )
            x_coord = [val - x_pia for val in x_coord]
            y_coord = [val - y_pia for val in y_coord]
            ax.scatter(x_coord, y_coord, color=color)
        plt.ylabel("y")
        plt.xlabel("x")
        # plt.savefig("bad_cell_1.png")
        plt.legend(["axons", "basal dendrites", "apical dendrites"])
        return ax

    def cortical_depth_hist(self, cell_id, cell_feat_orient_new_df):
        """
        Plot a histogram of the cortical depth of neurites for a given neuron.

        This method plots a histogram of the cortical depth of neurites for a specified neuron.
        The cortical depth is calculated with respect to the soma's distance from the pia.
        The resulting histogram provides insights into the distribution of neurites in the cortex.

        Parameters:
        -----------
        cell_id : int
            The ID of the neuron cell for which to plot the cortical depth histogram.
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons.

        Returns:
        --------
        ax : matplotlib.axes._axes.Axes
            The matplotlib axis containing the plotted cortical depth histogram.
        """

        fig, ax = plt.subplots(1, 1)
        morph = self.reconstruct(cell_id)
        morph_df = pd.DataFrame(morph.compartment_list)
        cell_idx = cell_feat_orient_new_df[
            cell_feat_orient_new_df["specimen_id"] == cell_id
        ].index
        slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
        upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
        shrink_factor = cell_feat_orient_new_df.loc[
            cell_idx, "estimated_shrinkage_factor"
        ].values

        x_soma, y_soma, z_soma = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df.loc[0, "x"],
            morph_df.loc[0, "y"],
            morph_df.loc[0, "z"],
            shrink_factor,
        )
        soma_distance_from_pia = cell_feat_orient_new_df.loc[
            cell_idx, "soma_distance_from_pia"
        ].values
        x_pia = x_soma
        y_pia = y_soma + soma_distance_from_pia
        x_coord, y_coord, z_coord = self.proper_rotation(
            slice_angle,
            upright_angle,
            morph_df["x"],
            morph_df["y"],
            morph_df["z"],
            shrink_factor,
        )
        x_coord = [val - x_pia for val in x_coord]
        y_coord = [val - y_pia for val in y_coord]
        ax.hist(
            np.array(y_coord).flatten(),
            orientation="horizontal",
            label="depth_from_pia",
            color="lightcoral",
        )
        plt.ylabel("depth")
        plt.xlabel("number")

    def _scatter_soma_position(self, cells_idx, cell_feat_orient_new_df):
        """
        Scatter plot the rotated soma positions of cells in the specified list.

        This method generates a scatter plot displaying the rotated soma positions of cells
        based on the given list of cell indices. The color of each point is determined by the layer
        type of the cell.

        Parameters:
        -----------
        cells_idx : list of int
            A list of cell indices (e.g., mice_spiny_orient_cells_idx) to plot.
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons.

        Returns:
        --------
        ax : matplotlib.axes._axes.Axes
            The matplotlib axis containing the plotted soma positions.
        """
        
        fig, ax = plt.subplots(1, 1)
        col_dict = {
            "1": "r",
            "2": "#ff7f0e",
            "2/3": "y",
            "3": "g",
            "4": "c",
            "5": "b",
            "6": "#9467bd",
            "6a": "#e377c2",
            "6b": "#7f7f7f",
        }

        for cell_idx in cells_idx:
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            color = col_dict[l_type]
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]

            # x_soma, y_soma, z_soma = self.proper_rotation(
            #     slice_angle,
            #     upright_angle,
            #     morph_df.loc[0, "x"],
            #     morph_df.loc[0, "y"],
            #     morph_df.loc[0, "z"],
            #     shrink_factor,
            # )
            # ax.scatter(x_soma, y_soma, color=color, label=color)

            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            x_soma = x_coord[0]
            y_soma = y_coord[0]
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ].values

            ax.scatter(x_coord[0], y_coord[0], color=color, label=color)

        red = mpatches.Patch(color="red", label="Layer 1")
        orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
        yellow = mpatches.Patch(color="y", label="Layer 2/3")
        green = mpatches.Patch(color="g", label="Layer 3")
        cian = mpatches.Patch(color="c", label="Layer 4")
        blue = mpatches.Patch(color="b", label="Layer 5")
        purple = mpatches.Patch(color="#9467bd", label="Layer 6")
        pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
        grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
        ax.invert_yaxis()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])

    def _soma_distance_hist(self, cell_feat_orient_new_df):
        """
        Plot a histogram of soma distances from the pia for all cells in cell_feat_orient_new_df.

        This method generates a histogram showing the distribution of soma distances from the pia
        for all cells available in the given DataFrame.

        Parameters:
        -----------
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons, including soma distances
            from the pia.

        Returns:
        --------
        ax : matplotlib.axes._axes.Axes
            The matplotlib axis containing the histogram plot.
        """
        
        fig, ax = plt.subplots(1, 1)
        plt.hist(
            cell_feat_orient_new_df.soma_distance_from_pia,
            orientation="horizontal",
            color="m",
        )
        ax.invert_yaxis()
        plt.ylabel("depth")
        plt.xlabel("number")

    def _soma_distance_hist_layer(self, cells_idx, cell_feat_orient_new_df):
        """
        Plot histograms of soma distances from the pia for cells in the cells_idx list.

        This method generates histograms showing the distribution of soma distances from the pia
        for cells in the specified cells_idx list. Each histogram is colored according to the layer type
        of the respective cells.

        Parameters:
        -----------
        cells_idx : list
            A list of cell indices for which to plot soma distance histograms.
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons, including soma distances
            from the pia and layer type information.
        
        Returns:
        --------
        data1, data2, data2_3, data3, data4, data5, data6, data6a, data6b : ndarray
            Arrays containing soma distances from the pia for each layer type.
            (e.g., data1 for Layer 1, data2 for Layer 2, etc.)
        """
        
        depth = []
        layer = []
        layer_color_dict = {
            "1": "r",
            "2": "#ff7f0e",
            "2/3": "y",
            "3": "g",
            "4": "c",
            "5": "b",
            "6": "#9467bd",
            "6a": "#e377c2",
            "6b": "#7f7f7f",
        }

        # Loop to get one layer at the time
        for cell_idx in cells_idx:
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            layer.append(layer_color_dict[l_type])
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ]
            depth.append(soma_distance_from_pia)

        depth_np = np.array(depth)
        layer_np = np.array(layer)

        fig, ax1 = plt.subplots(1, 1)

        # Loop to plot the histogram of the soma distance from the pia for each layer
        for l in layer_color_dict.values():
            # plt.hist(
            #     depth_np[layer_np == l],
            #     orientation="horizontal",
            #     color=l,
            #     label=l,
            #     alpha=0.5,
            # )
            if l == "r":
                data1 = depth_np[layer_np == l]
            if l == "#ff7f0e":
                data2 = depth_np[layer_np == l]
            if l == "y":
                data2_3 = depth_np[layer_np == l]
            if l == "g":
                data3 = depth_np[layer_np == l]
            if l == "c":
                data4 = depth_np[layer_np == l]
            if l == "b":
                data5 = depth_np[layer_np == l]
            if l == "#9467bd":
                data6 = depth_np[layer_np == l]
            if l == "#e377c2":
                data6a = depth_np[layer_np == l]
            if l == "#7f7f7f":
                data6b = depth_np[layer_np == l]
            sns.histplot(
                y=depth_np[layer_np == l],
                color=l,
                label=l,
                multiple="stack",
                kde=True,
            )

        fig, ax2 = plt.subplots(1, 1)
        for l in layer_color_dict.values():
            sns.histplot(
                y=depth_np[layer_np == l],
                color=l,
                label=l,
                element="poly",
            )


        red = mpatches.Patch(color="r", label="Layer 1")
        orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
        yellow = mpatches.Patch(color="y", label="Layer 2/3")
        green = mpatches.Patch(color="g", label="Layer 3")
        cian = mpatches.Patch(color="c", label="Layer 4")
        blue = mpatches.Patch(color="b", label="Layer 5")
        purple = mpatches.Patch(color="#9467bd", label="Layer 6")
        pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
        grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        plt.ylabel("soma_depth")
        plt.xlabel("x")
        plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])

        return data1, data2, data2_3, data3, data4, data5, data6, data6a, data6b

    def _soma_y_coord_hist_layer(self, cells_idx, cell_feat_orient_new_df):
        """
        Plot histograms of soma y-coordinate after rotation for cells in the cells_idx list.

        This method generates histograms showing the distribution of the soma y-coordinate
        after rotation for cells in the specified cells_idx list. Each histogram is colored
        according to the layer type of the respective cells.

        Parameters:
        -----------
        cells_idx : list
            A list of cell indices for which to plot soma y-coordinate histograms.
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons, including
            soma y-coordinates and layer type information.
        """
        
        soma_y_coord = []
        layer = []
        layer_color_dict = {
            "1": "r",
            "2": "#ff7f0e",
            "2/3": "y",
            "3": "g",
            "4": "c",
            "5": "b",
            "6": "#9467bd",
            "6a": "#e377c2",
            "6b": "#7f7f7f",
        }
        # Loop to get one layer at the time
        for cell_idx in cells_idx:
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]
            l_type = cell_feat_orient_new_df.loc[cell_idx, "structure__layer"]
            layer.append(layer_color_dict[l_type])
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            y_soma = y_coord[0]
            soma_y_coord.append(y_soma)

        soma_y_coord_np = np.array(soma_y_coord)
        layer_np = np.array(layer)

        fig, ax = plt.subplots(1, 1)

        # Loop to plot the histogram of the soma distance from the pia for each layer
        for l in layer_color_dict.values():
            plt.hist(
                soma_y_coord_np[layer_np == l],
                orientation="horizontal",
                color=l,
                label=l,
                alpha=0.5,
            )

        red = mpatches.Patch(color="r", label="Layer 1")
        orange = mpatches.Patch(color="#ff7f0e", label="Layer 2")
        yellow = mpatches.Patch(color="y", label="Layer 2/3")
        green = mpatches.Patch(color="g", label="Layer 3")
        cian = mpatches.Patch(color="c", label="Layer 4")
        blue = mpatches.Patch(color="b", label="Layer 5")
        purple = mpatches.Patch(color="#9467bd", label="Layer 6")
        pink = mpatches.Patch(color="#e377c2", label="Layer 6a")
        grey = mpatches.Patch(color="#7f7f7f", label="Layer 6b")
        ax.invert_yaxis()
        plt.ylabel("soma_depth")
        plt.xlabel("number")
        plt.legend(handles=[red, orange, yellow, green, cian, blue, purple, pink, grey])

    def _soma_y_coord_and_distance_scatter(self, cells_idx, cell_feat_orient_new_df):
        """
        Scatter plot of soma y-coordinate against soma distance from pia for selected cells.

        This method creates a scatter plot showing the relationship between soma y-coordinates,
        after rotation and adjustment, and soma distances from pia for a selected set of cells.

        Parameters:
        -----------
        cells_idx : list
            A list of cell indices for which to create the scatter plot.
        cell_feat_orient_new_df : pandas.DataFrame
            A DataFrame containing orientation-related features for neurons, including soma
            y-coordinates, soma distances from pia, and other relevant information.
        """

        fig, ax = plt.subplots(1, 1)
        depth = []
        soma_y_coord = []
        for cell_idx in cells_idx:
            soma_distance_from_pia = cell_feat_orient_new_df.loc[
                cell_idx, "soma_distance_from_pia"
            ]
            depth.append(soma_distance_from_pia)
            cell_id = cell_feat_orient_new_df.loc[cell_idx, "specimen_id"]
            morph = self.reconstruct(cell_id)
            morph_df = pd.DataFrame(morph.compartment_list)
            slice_angle = cell_feat_orient_new_df.loc[cell_idx, "estimated_slice_angle"]
            upright_angle = cell_feat_orient_new_df.loc[cell_idx, "upright_angle"]
            shrink_factor = cell_feat_orient_new_df.loc[
                cell_idx, "estimated_shrinkage_factor"
            ]
            x_coord, y_coord, z_coord = self.proper_rotation(
                slice_angle,
                upright_angle,
                morph_df["x"],
                morph_df["y"],
                morph_df["z"],
                shrink_factor,
            )
            y_pia = y_coord[0] + soma_distance_from_pia
            y_coord = [val - y_pia for val in y_coord]
            soma_y_coord.append(y_coord[0])
        soma_y_coord = [-y for y in soma_y_coord]
        ax.scatter(depth, soma_y_coord)

    def layer_boundaries(self, data1, data2_3, data4, data5, data6a, data6b):
        """
        Plot histogram of layer distribution and boundaries between layers for VISp mice cells.

        This method creates a histogram of soma distances from the pia for different cortical layers
        and identifies boundaries between these layers based on the distributions.

        Parameters:
        -----------
        (All of this data can be obtained from the _soma_distance_hist_layer method)
        data1 : array-like
            Soma distances from the pia for cells in Layer 1.
        data2_3 : array-like
            Soma distances from the pia for cells in Layers 2 and 3.
        data4 : array-like
            Soma distances from the pia for cells in Layer 4.
        data5 : array-like
            Soma distances from the pia for cells in Layer 5.
        data6a : array-like
            Soma distances from the pia for cells in Layer 6a.
        data6b : array-like
            Soma distances from the pia for cells in Layer 6b.

        Returns:
        --------
        tuple
            A tuple containing the soma distance boundaries between layers, in order:
            (Boundary 1, Boundary 2, Boundary 3, Boundary 4, Boundary 5).
        """

        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=data1,
            kde=True,
            color="red",
            orientation="horizontal",
            label="Distribution 1",
        )
        sns.histplot(
            data=data2_3,
            kde=True,
            color="y",
            orientation="horizontal",
            label="Distribution 2_3",
        )
        sns.histplot(
            data=data4,
            kde=True,
            color="c",
            orientation="horizontal",
            label="Distribution 4",
        )
        sns.histplot(
            data=data5,
            kde=True,
            color="b",
            orientation="horizontal",
            label="Distribution 5",
        )
        sns.histplot(
            data=data6a,
            kde=True,
            color="pink",
            orientation="horizontal",
            label="Distribution 6a",
        )
        sns.histplot(
            data=data6b,
            kde=True,
            color="grey",
            orientation="horizontal",
            label="Distribution 6b",
        )

        kde1 = stats.gaussian_kde(data1)
        kde2_3 = stats.gaussian_kde(data2_3)
        kde4 = stats.gaussian_kde(data4)
        kde5 = stats.gaussian_kde(data5)
        kde6a = stats.gaussian_kde(data6a)
        kde6b = stats.gaussian_kde(data6b)

        def kde_diff123(x):
            return kde1.evaluate(x) - kde2_3.evaluate(x)

        def kde_diff234(x):
            return kde2_3.evaluate(x) - kde4.evaluate(x)

        def kde_diff45(x):
            return kde4.evaluate(x) - kde5.evaluate(x)

        def kde_diff56a(x):
            return kde5.evaluate(x) - kde6a.evaluate(x)

        def kde_diff6ab(x):
            return kde6a.evaluate(x) - kde6b.evaluate(x)

        def find_intersection(func, x0, x1):
            return bisect(func, x0, x1)

        # Find the intersection points with binary search
        intersection_point_123 = find_intersection(
            kde_diff123,
            min(data1.min(), data2_3.min()),
            max(data1.max(), data2_3.max()),
        )
        intersection_point_234 = find_intersection(
            kde_diff234,
            min(data2_3.min(), data4.min()),
            max(data2_3.max(), data4.max()),
        )
        intersection_point_45 = find_intersection(
            kde_diff45, min(data4.min(), data5.min()), max(data4.max(), data5.max())
        )
        intersection_point_56a = find_intersection(
            kde_diff56a, min(data5.min(), data6a.min()), max(data5.max(), data6a.max())
        )
        intersection_point_6ab = find_intersection(
            kde_diff6ab,
            min(data6a.min(), data6b.min()),
            max(data6a.max(), data6b.max()),
        )

        # Plot vertical lines at the intersection points to get the boundaries
        plt.axvline(
            x=intersection_point_123, color="k", linestyle="--", label="Boundary 1"
        )
        plt.axvline(
            x=intersection_point_234, color="k", linestyle="--", label="Boundary 2"
        )
        plt.axvline(
            x=intersection_point_45, color="k", linestyle="--", label="Boundary 3"
        )
        plt.axvline(
            x=intersection_point_56a, color="k", linestyle="--", label="Boundary 4"
        )
        plt.axvline(
            x=intersection_point_6ab, color="k", linestyle="--", label="Boundary 5"
        )

        plt.legend()
        return (
            intersection_point_123,
            intersection_point_234,
            intersection_point_45,
            intersection_point_56a,
            intersection_point_6ab,
        )

    def plot_in_layer(self, cell_id, cell_feat_orient_new_df, cells_idx):
        """
        Plot the 2D morphology of a single neuron within its corresponding layer.

        This function generates a 2D plot of the morphology of a single neuron, positioned
        within its corresponding layer based on soma distance from the pia. The function
        utilizes the provided `cell_id`, `cell_feat_orient_new_df`, and `cells_idx` to
        determine the neuron's position within the layer and plots its morphology accordingly.
        Additionally, vertical dashed lines are drawn to represent layer boundaries.

        The soma distance from the pia for each cell in the `cells_idx` list is used to determine
        the position of the neuron within its layer, and the morphology is plotted accordingly.

        Parameters:
        -----------
        cell_id (int): The unique identifier of the neuron to be plotted (= specimen_id).

        cell_feat_orient_new_df (pd.DataFrame): DataFrame containing cell features and orientation data.

        cells_idx (list of int): List of cell indices for which to analyze and plot the morphology.

        Returns:
        -----------
        matplotlib.axes.Axes: The axes object representing the generated plot.
        """
        # I've called the following figures (1,2,3,4,5) to close them and avoid them being plotted
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        (
            data1,
            data2,
            data2_3,
            data3,
            data4,
            data5,
            data6,
            data6a,
            data6b,
        ) = self._soma_distance_hist_layer(cells_idx, cell_feat_orient_new_df)
        plt.close(fig1)
        plt.close(fig2)

        fig3 = plt.figure(3)
        fig4 = plt.figure(4)
        (
            intersection_point_123,
            intersection_point_234,
            intersection_point_45,
            intersection_point_56a,
            intersection_point_6ab,
        ) = self.layer_boundaries(data1, data2_3, data4, data5, data6a, data6b)
        plt.close(fig3)
        plt.close(fig4)

        fig5 = plt.figure(5)
        ax = self.plot_morphology_from_pia(cell_id, cell_feat_orient_new_df)
        # I want to change the sign of every intersection point
        intersection_point_123 = -intersection_point_123
        intersection_point_234 = -intersection_point_234
        intersection_point_45 = -intersection_point_45
        intersection_point_56a = -intersection_point_56a
        intersection_point_6ab = -intersection_point_6ab

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axhline(0, color="k", linestyle="--")
        ax.axhline(intersection_point_123, color="k", linestyle="--")
        ax.axhline(intersection_point_234, color="k", linestyle="--")
        ax.axhline(intersection_point_45, color="k", linestyle="--")
        ax.axhline(intersection_point_56a, color="k", linestyle="--")
        ax.axhline(intersection_point_6ab, color="k", linestyle="--")
        plt.close(fig5)
        plt.legend(["axons", "basal dendrites", "apical dendrites"])
