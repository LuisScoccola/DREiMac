from code import interact
from re import I
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, RadioButtons
from .geomtools import *
from .emcoords import *
from ripser import ripser
import warnings

# sparse symmetric matrix principal squareroot

sqrt_chebyshev_coeff = np.array([1.800632632, 0.6002108774, -0.1200421755, 0.05144664664, \
-0.02858147035, 0.01818820841, -0.01259183659, 0.009234013499, \
-0.007061304440, 0.005574714032, -0.004512863740, 0.003728017872, \
-0.003131535013, 0.002667603900, -0.002299658534, 0.002002928401, \
-0.001760149201])

def matrix_sqrt(A, degree=8):
    coeff = sqrt_chebyshev_coeff[:degree]
    s = A.shape[0]
    scale = scipy.sparse.linalg.norm(A)
    A = A * (1/scale) - scipy.sparse.identity(s)

    T = 2 * A
    d = scipy.sparse.identity(s) * coeff[-1]
    dd = scipy.sparse.csr_matrix((s,s))
    for n in range(coeff.shape[0]-2,0,-1):
        d, dd = T @ d - dd + coeff[n] * scipy.sparse.identity(s), d

    res = A @ d - dd + 0.5 * coeff[0] * scipy.sparse.identity(s)

    return res * np.sqrt(scale)


"""#########################################
        Main Circular Coordinates Class
#########################################"""
SCATTER_SIZE = 50

class CircularCoords(EMCoords):
    def __init__(self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "circ"

    def get_coordinates(self, perc = 0.5, inner_product = "uniform", cocycle_idxs = [[0]], normalize = True, partunity_fn = partunity_exp, return_gram_matrix=False):
        """
        Perform circular coordinates via persistent cohomology of 
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        inner_product : string
            Either 'uniform', 'exponential', or 'consistent'.
        cocycle_idxs : list of lists of integers
            Each list must consist of indices of possible cocycles, and represents the
            cohomology class given by adding the cocycles with the chosen indices
        normalize : bool
            Whether to return circular coordinates between 0 and 1
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        return_gram_matrix : boolean
            Whether to return the gram matrix consisting of the inner products
            between the selected cocycles
        """


        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        prime = self.prime_

        cocycles = []
        cohomdeaths = []
        cohombirths = []

        for cocycle_idx in cocycle_idxs:
            cohomdeath = -np.inf
            cohombirth = np.inf
            cocycle = np.zeros((0, 3))
            for k in range(len(cocycle_idx)):
                cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
                cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
                cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])
            cocycles.append(cocycle)
            cohomdeaths.append(cohomdeath)
            cohombirths.append(cohombirth)

        cohomdeath = min(cohomdeaths)
        cohombirth = max(cohombirths)


        ## Step 2: Determine radius for balls
        dist_land_data = self.dist_land_data_
        dist_land_land = self.dist_land_land_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        self.r_cover_ = r_cover # Store covering radius for reference
        if self.verbose:
            print("r_cover = %.3g"%r_cover)


        ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity 
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are %i point not covered by a landmark"%nzero)
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]

        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)
        

        threshold = 2*r_cover

        # boundary matrix
        neighbors = { i:set([]) for i in range(n_landmarks) }
        NEdges = n_landmarks**2
        edge_pair_to_index = {}
        l = 0
        row_index = []
        col_index = []
        value = []
        for i in range(n_landmarks):
            for j in range(n_landmarks):
                if i != j and dist_land_land[i,j] < threshold:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    edge_pair_to_index[(i,j)] = l
                    row_index.append(l)
                    col_index.append(i)
                    value.append(-1)
                    row_index.append(l)
                    col_index.append(j)
                    value.append(1)
                l += 1
        delta0 = sparse.coo_matrix((value, (row_index,col_index)),shape=(NEdges, n_landmarks)).tocsr()

        if inner_product=="uniform":
            row_index = []
            col_index = []
            value = []
            for l in edge_pair_to_index.values():
                row_index.append(l)
                col_index.append(l)
                value.append(1)
            WSqrt = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            W = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
        elif inner_product=="exponential":
            row_index = []
            col_index = []
            value = []
            sqrt_value = []
            for pl in edge_pair_to_index.items():
                p,l = pl
                i,j = p
                val = np.exp(-dist_land_land[i, j]**2/(threshold/2))
                row_index.append(l)
                col_index.append(l)
                value.append(val)
                sqrt_value.append(np.sqrt(val))
            WSqrt = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            W = scipy.sparse.coo_matrix((sqrt_value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
        elif inner_product=="consistent":
            #print("We assume dataset is Euclidean.")
            #print("start def inner product")
            row_index = []
            col_index = []
            value = []
            for i in range(n_landmarks):
                nn_i = np.argwhere(dist_land_data[i,:] < threshold)[:,0]
                n_nn_i = len(nn_i)
                if n_nn_i == 0:
                    continue
                [aa, bb] = np.meshgrid(nn_i, nn_i)
                aa = aa[np.triu_indices(n_nn_i, 1)]
                bb = bb[np.triu_indices(n_nn_i, 1)]
                how_many = 5
                idx = np.argsort( np.linalg.norm(self.X_[aa] - self.X_[bb], axis=1)  )[:how_many]
                #fraction = 8
                #idx = np.arange(len(aa))
                #idx = idx[np.linalg.norm(self.X_[aa] - self.X_[bb], axis=1) < (r_cover/fraction)]
                aa = aa[idx]
                bb = bb[idx]
                K = lambda x,y : np.exp( - (np.linalg.norm(self.X_[x] - self.X_[y], axis=1) / (r_cover))**2 )
                partial_prod = (varphi[i,aa] + varphi[i,bb]) * K(aa,bb)
                #print(partial_prod[partial_prod>0])
                #print( K(aa,bb).shape )
                for j in neighbors[i]:
                    for k in neighbors[i]:
                        if dist_land_land[j,k] >= threshold :
                            continue
                        a = edge_pair_to_index[(i,j)]
                        b = edge_pair_to_index[(i,k)]
                        row_index.append(a)
                        col_index.append(b)
                        val = 2 * np.sum( partial_prod * (varphi[j,bb] - varphi[j,aa]) * (varphi[k,bb] - varphi[k,aa]))
                        value.append(val)
            W = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            #print("defined inner product")

            WSqrt = matrix_sqrt(W, degree=8)
            #print("took square root")
        else:
            raise Exception("inner_product must be uniform, exponential, or consistent!")

        A = WSqrt @ delta0

        all_tau = []

        Ys = []

        for cocycle in cocycles:
            ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
            ## and use it to find a projection of the cocycle
            ## onto the image of delta0

            Y = np.zeros((NEdges,))
            for i, j, val in cocycle:
                # lift to integer cocycle
                if val > (prime-1)/2:
                    val -= prime
                if (i,j) in edge_pair_to_index:
                    Y[edge_pair_to_index[(i,j)]] = val
                    Y[edge_pair_to_index[(j,i)]] = -val
 
            b = WSqrt.dot(Y)
            tau = lsqr(A, b)[0]

            Y = Y - delta0.dot(tau)
            Ys.append(Y)

            all_tau.append(tau)

        gram_matrix = np.zeros((len(Ys),len(Ys)))
        for i in range(len(Ys)):
            for j in range(len(Ys)):
                gram_matrix[i,j] = Ys[i].T @ W @ Ys[j]

        circ_coords = []
        for Y, tau in zip(Ys,all_tau):
            ## Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        
            # compute all transition functions
            theta_matrix = np.zeros((n_landmarks, n_landmarks))

            for pl in edge_pair_to_index.items():
                p,l = pl
                i,j = p
                v = np.mod(Y[l] + 0.5, 1) - 0.5
                theta_matrix[i, j] = v
            class_map = -tau[ball_indx]
            for i in range(n_data):
                class_map[i] += theta_matrix[ball_indx[i], :].dot(varphi[:, i])
            thetas = np.mod(2*np.pi*class_map, 2*np.pi)
            circ_coords.append(thetas)

        if normalize:
            circ_coords = np.array(circ_coords) / (2 * np.pi)

        if return_gram_matrix:
            return circ_coords, gram_matrix
        else:
            return circ_coords

    def update_colors(self):
        if len(self.selected) > 0:
            idxs = np.array(list(self.selected))
            self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
            ## Step 2: Update circular coordinates on point cloud
            thetas = self.coords
            c = plt.get_cmap('magma_r')
            thetas -= np.min(thetas)
            thetas /= np.max(thetas)
            thetas = np.array(np.round(thetas*255), dtype=int)
            C = c(thetas)
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color(C)
            else:
                self.coords_scatter._facecolor3d = C
                self.coords_scatter._edgecolor3d = C
        else:
            self.selected_plot.set_offsets(np.zeros((0, 2)))
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color('C0')
            else:
                self.coords_scatter._facecolor3d = 'C0'
                self.coords_scatter._edgecolor3d = 'C0'

    def recompute_coords_dimred(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        colors accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        self.update_colors()
        
    def onpick_dimred(self, evt):
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_dimred(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.ax_coords.figure.canvas.draw()
        return True

    def on_perc_slider_move_dimred(self, evt):
        self.recompute_coords_dimred()

    def on_partunity_selector_change_dimred(self, evt):
        self.recompute_coords_dimred()

    def plot_dimreduced(self, Y, using_jupyter = True, init_params = {'cocycle_idxs':[], 'perc':0.99, 'partunity_fn':partunity_linear, 'azim':-60, 'elev':30}, dpi=None):
        """
        Do an interactive plot of circular coordinates, coloring a dimension
        reduced version of the point cloud by the circular coordinates

        Parameters
        ----------
        Y: ndarray(N, d)
            A 2D point cloud with the same number of points as X
        using_jupyter: boolean
            Whether this is an interactive plot in jupyter
        init_params: dict
            The intial parameters.  Optional fields of the dictionary are as follows:
            {
                cocycle_idxs: list of int
                    A list of cocycles to start with
                u: ndarray(3, float)
                    The initial stereographic north pole
                perc: float
                    The percent coverage to start with
                partunity_fn: (dist_land_data, r_cover) -> phi
                    The partition of unity function to start with
                azim: float
                    Initial azimuth for 3d plots
                elev: float
                    Initial elevation for 3d plots
            }
        dpi: int
            Dot pixels per inch
        """
        if Y.shape[1] < 2 or Y.shape[1] > 3:
            raise Exception("Dimension reduced version must be in 2D or 3D")
        self.Y = Y
        if using_jupyter and in_notebook():
            import matplotlib
            matplotlib.use("nbAgg")
        if not dpi:
            dpi = compute_dpi(2, 1)
        fig = plt.figure(figsize=(DREIMAC_FIG_RES*2, DREIMAC_FIG_RES), dpi=dpi)
        ## Step 1: Plot H1
        self.ax_persistence = fig.add_subplot(121)
        self.setup_ax_persistence(y_compress=1.37)
        fig.canvas.mpl_connect('pick_event', self.onpick_dimred)
        self.selected = set([])

        ## Step 2: Setup window for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle
        self.perc_slider, self.partunity_selector, self.selected_cocycle_text, _ = EMCoords.setup_param_chooser_gui(self, fig, 0.25, 0.75, 0.4, 0.5, init_params)
        self.perc_slider.on_changed(self.on_perc_slider_move_dimred)
        self.partunity_selector.on_clicked(self.on_partunity_selector_change_dimred)

        ## Step 3: Setup axis for coordinates
        if Y.shape[1] == 3:
            self.ax_coords = fig.add_subplot(122, projection='3d')
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=SCATTER_SIZE, cmap='magma_r')
            set_3dplot_equalaspect(self.ax_coords, Y)
            if 'azim' in init_params:
                self.ax_coords.azim = init_params['azim']
            if 'elev' in init_params:
                self.ax_coords.elev = init_params['elev']
        else:
            self.ax_coords = fig.add_subplot(122)
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], s=SCATTER_SIZE, cmap='magma_r')
            self.ax_coords.set_aspect('equal')
        self.ax_coords.set_title("Dimension Reduced Point Cloud")
        if len(init_params['cocycle_idxs']) > 0:
            # If some initial cocycle indices were chosen, update
            # the plot
            self.recompute_coords_dimred(init_params['cocycle_idxs'])
        plt.show()
    
    def get_selected_dimreduced_info(self):
        """
        Return information about what the user selected and their viewpoint in
        the interactive dimension reduced plot

        Returns
        -------
        {
            'partunity_fn': (dist_land_data, r_cover) -> phi
                The selected function handle for the partition of unity
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'perc': float
                The selected percent coverage,
            'azim':float
                Azumith if viewing in 3D
            'elev':float
                Elevation if viewing in 3D
        }
        """
        ret = EMCoords.get_selected_info(self)
        if self.Y.shape[1] == 3:
            ret['azim'] = self.ax_coords.azim
            ret['elev'] = self.ax_coords.elev
        return ret

    def update_plot_torii(self, circ_idx):
        """
        Update a joint plot of circular coordinates, switching between
        2D and 3D modes if necessary

        Parameters
        ----------
        circ_idx: int
            Index of the circular coordinates that have
            been updated
        """
        N = self.plots_in_one
        n_plots = len(self.plots)
        ## Step 1: Figure out the index of the involved plot
        plot_idx = int(np.floor(circ_idx/N))
        plot = self.plots[plot_idx]

        ## Step 2: Extract the circular coordinates from all
        ## plots that have at least one cochain representative selected
        labels = []
        coords = []
        for i in range(N):
            idx = plot_idx*N + i
            c_info = self.coords_info[idx]
            if len(c_info['selected']) > 0:
                # Only include circular coordinates that have at least
                # one persistence dot selected
                coords.append(c_info['coords'])
                labels.append("Coords {}".format(idx))
        ## Step 3: Adjust the plot accordingly
        if len(labels) > 0:
            X = np.array([])
            if len(labels) == 1:
                # Just a single coordinate; put it on a circle
                coords = np.array(coords).flatten()
                X = np.array([np.cos(coords), np.sin(coords)]).T
            else:
                X = np.array(coords).T
            updating_axes = False
            if X.shape[1] == 3 and plot['axis_2d']:
                # Need to switch from 2D to 3D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = False
                updating_axes = True
            elif X.shape[1] == 2 and not plot['axis_2d']:
                # Need to switch from 3D to 2D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = True
                updating_axes = True
            if X.shape[1] == 3:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx, projection='3d')
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], X[:, 2], s=SCATTER_SIZE, c=self.coords_colors)
                    plot['ax'].set_title('Joint 3D Plot')
                else:
                    plot['coords_scatter'].set_offsets(X)
                set_pi_axis_labels(plot['ax'], labels)
            else:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx)
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], s=SCATTER_SIZE, c=self.coords_colors)
                else:
                    plot['coords_scatter'].set_offsets(X)
                if len(labels) > 1:
                    set_pi_axis_labels(plot['ax'], labels)
                    plot['ax'].set_title('Joint 2D Plot')
                else:
                    plot['ax'].set_xlabel('')
                    plot['ax'].set_xlim([-1.1, 1.1])
                    plot['ax'].set_ylabel('')
                    plot['ax'].set_ylim([-1.1, 1.1])
                    plot['ax'].set_title(labels[0])
        else:
            X = np.array([])
            if plot['axis_2d']:
                X = -2*np.ones((self.X_.shape[0], 2))
            else:
                X = -2*np.ones((self.X_.shape[0], 3))
            plot['coords_scatter'].set_offsets(X)
            
    
    def recompute_coords_torii(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        joint torii plots accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        # Save away circular coordinates
        self.coords_info[self.selected_coord_idx]['selected'] = self.selected
        self.coords_info[self.selected_coord_idx]['coords'] = self.coords
        self.update_plot_torii(self.selected_coord_idx)

    def onpick_torii(self, evt):
        """
        Handle a pick even for the torii plot
        """
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_torii(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.fig.canvas.draw()
        return True

    def select_torii_coord(self, idx):
        """
        Select a particular circular coordinate plot and un-select others
        
        Parameters
        ----------
        idx: int
            Index of the plot to select
        """
        for i, coordsi in enumerate(self.coords_info):
            if i == idx:
                self.selected_coord_idx = idx
                coordsi = self.coords_info[idx]
                # Swap in the appropriate GUI objects for selection
                self.selected = coordsi['selected']
                self.selected_cocycle_text = coordsi['selected_cocycle_text']
                self.perc_slider = coordsi['perc_slider']
                self.partunity_selector = coordsi['partunity_selector']
                self.persistence_text_labels = coordsi['persistence_text_labels']
                self.coords = coordsi['coords']
                coordsi['button'].color = 'red'
                for j in np.array(list(self.selected)):
                    self.persistence_text_labels[j].set_text("%i"%j)
                idxs = np.array(list(self.selected), dtype=int)
                if idxs.size > 0:
                    self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
                else:
                    self.selected_plot.set_offsets(np.array([[np.nan]*2]))
            else:
                coordsi['button'].color = 'gray'
        self.ax_persistence.set_title("H1 Cocycle Selection: Coordinate {}".format(idx))

    def on_perc_slider_move_torii(self, evt, idx):
        """
        React to a change in coverage
        a particular circular coordinate, and recompute the 
        coordinates if they aren't trivial
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)
        if len(self.selected) > 0:
            self.recompute_coords_torii()

    def on_partunity_selector_change_torii(self, evt, idx):
        """
        React to a change in partition of unity type for 
        a particular circular coordinate, and recompute the 
        coordinates if they aren't trivial
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)
        if len(self.selected) > 0:
            self.recompute_coords_torii()

    def on_click_torii_button(self, evt, idx):
        """
        React to a click event, and change the selected
        circular coordinate if necessary
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)

    def plot_torii(self, f, using_jupyter=True, zoom=1, dpi=None, coords_info=2, plots_in_one = 2, lowerleft_plot = None, lowerleft_3d=False):
        """
        Do an interactive plot of circular coordinates, where points are drawn on S1, 
        on S1 x S1, or S1 x S1 x S1

        Parameters
        ----------
        f: Display information for the points
            On of three options:
            1) A scalar function with which to color the points, represented
               as a 1D array
            2) A list of colors with which to color the points, specified as
               an Nx3 array
            3) A list of images to place at each location
        using_jupyter: boolean
            Whether this is an interactive plot in jupyter
        zoom: int
            If using patches, the factor by which to zoom in on them
        dpi: int
            Dot pixels per inch
        coords_info: Information about how to perform circular coordinates.  There will
            be as many plots as the ceil of the number of circular coordinates, and
            they will be plotted pairwise.
            This parameter is one of two options
            1) An int specifying the number of different circular coordinate
               functions to compute
            2) A list of dictionaries with pre-specified initial parameters for
               each circular coordinate.  Each dictionary has the following keys:
               {
                    'cocycle_reps': dictionary
                        A dictionary of cocycle representatives, with the key
                        as the cocycle index, and the value as the coefficient
                    TODO: Finish update to support this instead of a set
                    'perc': float
                        The percent coverage to start with,
                    'partunity_fn': (dist_land_data, r_cover) -> phi
                        The partition of unity function to start with
               }
        plots_in_one: int
            The max number of circular coordinates to put in one plot
        lowerleft_plot: function(matplotlib axis)
            A function that plots something in the lower left
        lowerleft_3d: boolean
            Whether the lower left plot is 3D
        """
        if plots_in_one < 2 or plots_in_one > 3:
            raise Exception("Cannot be fewer than 2 or more than 3 circular coordinates in one plot")
        self.plots_in_one = plots_in_one
        self.f = f
        ## Step 1: Figure out how many plots are needed to accommodate all
        ## circular coordinates
        n_plots = 1
        if type(coords_info) is int:
            n_plots = int(np.ceil(coords_info/plots_in_one))
            coords_info = []
        else:
            n_plots = int(np.ceil(len(coords_info)/plots_in_one))
        while len(coords_info) < n_plots*plots_in_one:
            coords_info.append({'selected':set([]), 'perc':0.99, 'partunity_fn':partunity_linear})
        self.selecting_idx = 0 # Index of circular coordinate which is currently being selected
        if using_jupyter and in_notebook():
            import matplotlib
            matplotlib.use("nbAgg")
        if not dpi:
            dpi = compute_dpi(n_plots+1, 2)
        fig = plt.figure(figsize=(DREIMAC_FIG_RES*(n_plots+1), DREIMAC_FIG_RES*2), dpi=dpi)
        self.dpi = dpi
        self.fig = fig

        ## Step 2: Setup H1 plot, along with initially empty text labels
        ## for each persistence point
        self.ax_persistence = fig.add_subplot(2, n_plots+1, 1)
        self.setup_ax_persistence()
        fig.canvas.mpl_connect('pick_event', self.onpick_torii)


        ## Step 2: Setup windows for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle for each circular coordinate.
        ## Also store variables for selecting cocycle representatives
        width = 1/(n_plots+1)
        height = 1/plots_in_one
        partunity_keys = tuple(PARTUNITY_FNS.keys())
        for i in range(n_plots):
            xstart = width*(i+1.4)
            for j in range(plots_in_one):
                idx = i*plots_in_one+j
                # Setup plots and state for a particular circular coordinate
                ystart = 0.8 - 0.4*height*j
                coords_info[idx]['perc_slider'], coords_info[idx]['partunity_selector'], coords_info[idx]['selected_cocycle_text'], coords_info[idx]['button'] = self.setup_param_chooser_gui(fig, xstart, ystart, width, height, coords_info[idx], idx)
                coords_info[idx]['perc_slider'].on_changed(callback_factory(self.on_perc_slider_move_torii, idx))
                coords_info[idx]['partunity_selector'].on_clicked = callback_factory(self.on_partunity_selector_change_torii, idx)
                coords_info[idx]['button'].on_clicked(callback_factory(self.on_click_torii_button, idx))
                dgm = self.dgm1_lifetime
                coords_info[idx]['persistence_text_labels'] = [self.ax_persistence.text(dgm[i, 0], dgm[i, 1], '') for i in range(dgm.shape[0])]
                coords_info[idx]['idx'] = idx
                coords_info[idx]['coords'] = np.zeros(self.X_.shape[0])
        self.coords_info = coords_info

        ## Step 3: Figure out colors of coordinates
        self.coords_colors = None
        if not (type(f) is list):
            # Figure out colormap if images aren't passed along
            self.coords_colors = f
            if f.size == self.X_.shape[0]:
                # Scalar function, so need to apply colormap
                c = plt.get_cmap('magma_r')
                fscaled = f - np.min(f)
                fscaled = fscaled/np.max(fscaled)
                C = c(np.array(np.round(fscaled*255), dtype=np.int32))
                self.coords_colors = C[:, 0:3]
        
        ## Step 4: Setup plots
        plots = []
        self.n_plots = n_plots
        for i in range(n_plots):
            # 2D by default, but can change to 3D later
            ax = fig.add_subplot(2, n_plots+1, n_plots+3+i)
            pix = -2*np.ones(self.X_.shape[0])
            plot = {}
            plot['ax'] = ax
            plot['coords_scatter'] = ax.scatter(pix, pix, s=SCATTER_SIZE, c=self.coords_colors) # Scatterplot for circular coordinates
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            plot['axis_2d'] = True
            plot['patch_boxes'] = [] # Array of image patch display objects
            plots.append(plot)
        self.plots = plots

        ## Step 5: Initialize plots with information passed along
        for i in reversed(range(len(coords_info))):
            self.select_torii_coord(i)
            self.recompute_coords_torii([])
        
        ## Step 6: Plot something in the lower left corner if desired
        if lowerleft_plot:
            if lowerleft_3d:
                ax = fig.add_subplot(2, n_plots+1, n_plots+2, projection='3d')
            else:
                ax = fig.add_subplot(2, n_plots+1, n_plots+2)
            lowerleft_plot(ax)

        plt.show()

def do_two_circle_test():
    """
    Test interactive plotting with two noisy circles of different sizes
    """
    prime = 41
    np.random.seed(2)
    N = 500
    X = np.zeros((N*2, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    X[N::, 0] = 2*np.cos(t) + 4
    X[N::, 1] = 2*np.sin(t) + 4
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    X = X + 0.2*np.random.randn(X.shape[0], 2)

    f = np.concatenate((t, t + np.max(t)))
    f = f[perm]
    fscaled = f - np.min(f)
    fscaled = fscaled/np.max(fscaled)
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(fscaled*255), dtype=np.int32))[:, 0:3]
    #plt.scatter(X[:, 0], X[:, 1], s=SCATTER_SIZE, c=C)
    
    cc = CircularCoords(X, 100, prime = prime)
    #cc.plot_dimreduced(X, using_jupyter=False)
    cc.plot_torii(f, coords_info=2, plots_in_one=3)

def do_torus_test():
    """
    Test interactive plotting with a torus
    """
    prime = 41
    np.random.seed(2)
    N = 10000
    R = 5
    r = 2
    X = np.zeros((N, 3))
    s = np.random.rand(N)*2*np.pi
    t = np.random.rand(N)*2*np.pi
    X[:, 0] = (R + r*np.cos(s))*np.cos(t)
    X[:, 1] = (R + r*np.cos(s))*np.sin(t)
    X[:, 2] = r*np.sin(s)

    cc = CircularCoords(X, 100, prime=prime)
    f = s
    def plot_torus(ax):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=f, cmap='magma_r')
        set_3dplot_equalaspect(ax, X)

    cc.plot_torii(f, coords_info=2, plots_in_one=2, lowerleft_plot=plot_torus, lowerleft_3d=True)



### Circular Coordinates AUX
from sklearn.manifold import Isomap
from sklearn.neighbors import KDTree

# operations with circular coordinates
def center_circ_coord(coord, bins=50):
    coord = coord.copy()
    
    x = np.hstack(coord)
    vals, ticks = np.histogram(x,bins=bins)
    coord = ((coord - ticks[np.argmax(vals)]) + 0.5) % 1
    return coord

def sum_circ_coords(c1, c2):
    return (c1 + c2) % 1

def sub_circ_coords(c1, c2):
    return (c1 - c2) % 1

def offset_circ_coord(c, offset):
    return (c + offset) % 1


# improve circular coordinates with lattice reduction
def reduce_circular_coordinates(circ_coords, gram_matrix):
    lattice_red_input = np.linalg.cholesky(gram_matrix)
    new_vectors, change_basis = LLL(lattice_red_input.T)
    change_basis = change_basis.T
    new_gram_matrix = new_vectors.T @ new_vectors
    new_circ_coords = (change_basis @ circ_coords) % 1
    return new_circ_coords, new_gram_matrix, change_basis

# approximate geodesic distance
def geodesic_distance(X, k = 15):
    iso = Isomap(n_components = 2,n_neighbors=k)
    return iso.fit(X).dist_matrix_


# sliding window embedding
def sw(ts, d, tau):
    emb = []
    last = len(ts) - ((d - 1) * tau)
    for i in range(last):
        emb.append(ts[i:i + d * tau:tau])
    return np.array(emb)


# Dirichlet form between maps to circle
exp_weight = lambda radius : lambda d : np.exp(-d**2/radius**2)
const_weight = lambda d : 1

def differential_circle_valued_map(fi,fj):
    if fi < fj:
        a = fj - fi
        b = (fj - 1) - fi
    else:
        a = fj - fi
        b = (fj + 1) - fi
    if np.abs(a) < np.abs(b):
        return a
    else:
        return b

def dirichlet_form(X,f,g,weight,eps,k,graph_type):
    ##print("We assume dataset is Euclidean.")
    # type may be "k" or "eps"
    # f and g are vectors of numbers between 0 and 1, interpreted as maps to the circle
    # we assume Euclidean distance
    tree = KDTree(X)
    if graph_type=="k":
        _, neighbors = tree.query(X,k=k)
    if graph_type=="eps":
        neighbors = tree.query_radius(X,r=eps)
    checked = set()
    partial_form = 0
    for i in range(X.shape[0]):
        for j in neighbors[i]:
            if (i,j) in checked or (j,i) in checked:
                continue
            dist_ij = np.linalg.norm(X[i] - X[j])
            df_ij = differential_circle_valued_map(f[i], f[j])
            dg_ij = differential_circle_valued_map(g[i], g[j])
            partial_form += weight(dist_ij) * df_ij * dg_ij
    return partial_form

def dirichlet_form_gram_matrix(X,circ_coords,weight,eps=None,k=None,graph_type="k"):
    number_circ_coords = len(circ_coords)
    res = np.zeros((number_circ_coords,number_circ_coords))
    for i in range(number_circ_coords):
        for j in range(number_circ_coords):
            x = dirichlet_form(X,circ_coords[i], circ_coords[j], weight, eps, k, graph_type)
            res[i,j] = x
            res[j,i] = x
    return res


# prominence diagram
def prominence_diagram(dgm, max_proms=10):
    prominences = dgm[:,1] - dgm[:,0]
    how_many = min(max_proms, len(prominences))
    order_by_prominence = np.argsort(-prominences)
    prominences = prominences[order_by_prominence]
    return order_by_prominence


#### Lattice Reduction

# Optimal algorithm for the case of two vectors:

def Lagrange(B):
    B = B.copy()
    change = np.eye(B.shape[0])

    if np.linalg.norm(B[:,0]) < np.linalg.norm(B[:,1]):
        B0 = B[:,0].copy()
        B1 = B[:,1].copy()
        B[:,0], B[:,1] = B1, B0
        change0 = change[:,0].copy()
        change1 = change[:,1].copy()
        change[:,0], change[:,1] = change1, change0

    
    while np.linalg.norm(B[:,1]) <= np.linalg.norm(B[:,0]):
        q = round(np.dot(B[:,0],B[:,1]/np.linalg.norm(B[:,1])**2))
        r = B[:,0] - q * B[:,1]
        c = change[:,0] - q * change[:,1]
        B[:,0] = B[:,1]
        change[:,0] = change[:,1]
        B[:,1] = r
        change[:,1] = c

    return B, change
    

#Gram-Schmidt (without normalization)

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

# projects v2 onto v1
def proj(v1, v2):
    return gs_cofficient(v1, v2) * v1


def GS(B):
    n = len(B)
    A = np.zeros((n,n))
    A[:,0] = B[:,0]
    for i in range(1,n):
        Ai = B[:, i]
        for j in range(0, i):
            Aj = B[:, j]
            #t = np.dot(B[i],B[j])
            Ai = Ai - proj(Aj,Ai)
        A[:, i] = Ai
    return A

# LLL algorithm 

def LLL(B, delta = 3/4):
    B = B.copy()
    Q = GS(B)
    change = np.eye(B.shape[0])

    def mu(i,j):
        v = B[:,i]
        u = Q[:,j]
        return np.dot(v,u) / np.dot(u,u)   

    n, k = len(B), 1
    while k < n:

        # length reduction step
        for j in reversed(range(k)):
            if abs(mu(k,j)) > .5:
                mu_kj = mu(k,j)
                B[:,k] = B[:,k] - round(mu_kj)*B[:,j]
                change[:,k] = change[:,k] - round(mu_kj)*change[:,j]
                Q = GS(B)

        # swap step
        if np.dot(Q[:,k],Q[:,k]) > (delta - mu(k,k-1)**2)*(np.dot(Q[:,k-1],Q[:,k-1])):
            k = k + 1
        else:
            B_k = B[:,k].copy()
            B_k1 = B[:,k-1].copy()
            B[:,k], B[:,k-1] = B_k1, B_k
            change_k = change[:,k].copy()
            change_k1 = change[:,k-1].copy()
            change[:,k], change[:,k-1] = change_k1, change_k
 
            Q = GS(B)
            k = max(k-1, 1)

    return B, change

