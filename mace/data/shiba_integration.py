###########################################################################################
# Integrating pickle files processing for Shiba's glass dataset
# Authors: Luca Teodorescu
###########################################################################################

import torch
import glob
import pickle
import numpy as np
import scipy.spatial
from torch_geometric.data import Data, Dataset
from torch_scatter         import scatter
from typing                import Sequence


def get_targets(initial_positions: np.ndarray,trajectory_target_positions: Sequence[np.ndarray]) -> np.ndarray:
  """Returns the averaged particle mobilities from the sampled trajectories.

  Args:
    initial_positions: the initial positions of the particles with shape
      [n_particles, 3].
    trajectory_target_positions: the absolute positions of the particles at the
      target time for all sampled trajectories, each with shape
      [n_particles, 3].
  """
  targets = np.mean([np.linalg.norm(t - initial_positions, axis=-1)
                     for t in trajectory_target_positions]
                     , axis=0)
  return targets.astype(np.float32)

class ShibaDataset_large(Dataset):
    def __init__(self, root, listTimesIdx, IS, e_pot, r_c, set_type, N_train=400, transform=None, pre_transform=None, pre_filter=None):

        self.input_dir = root
        self.listTimesIdx  = listTimesIdx
        self.set_type = set_type
        self.N_train  = N_train
        self.IS       = IS
        self.e_pot    = e_pot
        self.r_c      = r_c
        self.pot_th   = 2.5  ## not an aenerggy, but the lenght of the Energy Potential used in the MD.

        self.train_idxs = np.arange(1,401)      ##according to Shiba train-test split
        self.test_idxs  = np.arange(401, 501)

        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        raw_files = glob.glob(self.input_dir + '/raw/*')
        if self.set_type == 'train':
            raw_files = [name.split('/')[-1] for name in raw_files if int(name.split('/')[-1].split('_')[-2]) in self.train_idxs]
            ##selecting limited number of sample
            if self.N_train > len(self.train_idxs):
                raise Exception("N_train larger than train set size")
            raw_files = raw_files[:self.N_train]
        elif self.set_type == 'test':
            raw_files = [name.split('/')[-1] for name in raw_files if int(name.split('/')[-1].split('_')[-2]) in self.test_idxs]
        else:
            raise Exception('Not allowed type')

        return raw_files

    @property
    def processed_file_names(self):
        return [file[:-4]+'.pt' for file in self.raw_file_names]

    def download(self):
        pass

    def calc_stats(self):
        ys = np.array([data.y.numpy() for data in self])         ##node targets size [400, 4096, 10]
        types = self[0].x
        Ntargets= ys.shape[2]

        Ntypes = types.max()+1
        mean_dev_per_type = np.zeros( (Ntypes, Ntargets, 2))
        mean_dev_epot     = np.zeros(2)
        mean_dev_deltaR   = np.zeros(2)

        for part_type in range(Ntypes):
            selected_particles = np.where(types == part_type)[0]
            mean_dev_per_type[part_type, :, 0] = np.mean(ys[:,selected_particles,:], axis=(0,1))
            mean_dev_per_type[part_type, :, 1] = np.std( ys[:,selected_particles,:], axis=(0,1))

        epots = np.array([data.e_pot.numpy() for data in self])   ##size [400, 4096]
        mean_dev_epot[0] = np.mean(epots)
        mean_dev_epot[1] = np.std(epots)

        deltaRs = np.array([data.delta_r_cage.numpy() for data in self])  ##size [400, 4096]
        mean_dev_deltaR[0] = np.mean(deltaRs)
        mean_dev_deltaR[1] = np.std(deltaRs)

        return mean_dev_per_type, mean_dev_epot, mean_dev_deltaR


    def process(self):
        # Read data into huge `Data` list.
        for ind,path in enumerate(self.raw_paths):
            ## Uploading thermal and IS data (same final positions but different initial ones)
            with open(path, 'rb') as f:
                data = np.load(f)

                types            = data['types'].astype(np.int32)
                positions_th     = data['positions'].astype(np.float32)
                positions_IS     = data['positions_IS'].astype(np.float32)
                times            = data['times'].astype(np.int32)
                box              = data['box']
                target_positions = data['trajectory_target_positions']

            ## number of targets
            n_idx = len(target_positions)

            ## targets for all the timescales
            targets = []
            for timIdx in range(n_idx):
                targets.append(get_targets(positions_th, target_positions[timIdx]))
            targets = np.array(targets)

            Lx    = box[0]

            edge_index_list   = []
            edge_attr_list    = []
            e_pot_list        = []
            pair_pot_list     = []
            edge_targets_list = []

            for i, positions in enumerate((positions_th, positions_IS)):
                ## Generate the graph
                distance_upper_bound = np.array([self.pot_th, self.r_c]).max()
                tree = scipy.spatial.cKDTree(positions, boxsize=Lx + 1e-8)                               ##using PBC
                _, col = tree.query(positions, k=4096,distance_upper_bound=distance_upper_bound + 1e-8)  ##max number of neighbors == all the particles
                col = col[:, 1:]                                                                         ##receivers padded
                row = np.array([np.ones(len(c)) * i for (i, c) in enumerate(col)], dtype=int)            ##senders padded
                cf = col.flatten()                                                                       ##flatten version
                rf = row.flatten()
                mask = cf < tree.n                                                                       ##remove padding

                edge_index = np.array([rf[mask], cf[mask]], dtype=int)
                edge_attr = positions[edge_index[1]] - positions[edge_index[0]]

                ## Enforcing PBC also to relative positions
                edge_attr[(edge_attr > Lx / 2.0)] -= Lx
                edge_attr[(edge_attr < -Lx / 2.0)] += Lx
                edge_norms = np.sum(edge_attr ** 2, axis=-1) ** (1 / 2)

                ## Epot computation
                energy_scales = np.array([1, 1.5, 0.5])                   ##epsilon_AA, epsilon_AB, epsilon_BB
                sigmas = np.array([1, 0.8, 0.88])                         ##sigma_AA  , sigma_AB  , sigma_BB
                edge_types = types[edge_index[0]] + types[edge_index[1]]  ## 0 = AA, 1 = AB, 2 = BB

                edge_epsilon = energy_scales[edge_types]
                edge_sigmas = sigmas[edge_types]
                pairwise_potentials = 4 * edge_epsilon * (
                            (edge_sigmas / edge_norms) ** 12 - (edge_sigmas / edge_norms) ** 6)
                if distance_upper_bound > self.pot_th:
                    pairwise_potentials[edge_norms > self.pot_th] = 0.  ##setting to 0 interaction larger than potential cutoff

                e_pot_tensor = scatter(torch.tensor(pairwise_potentials), torch.tensor(edge_index[0]),dim=0)  ##summing over neighborhoods: using pytorch_scatter function out of lazyness :)
                e_pot        = np.array(e_pot_tensor)

                if distance_upper_bound > self.r_c:
                    ## Restricting edges to d_th
                    edge_index = edge_index[:, edge_norms <= self.r_c]
                    edge_attr = edge_attr[edge_norms <= self.r_c]
                    pairwise_potentials = pairwise_potentials[edge_norms <= self.r_c]

                # computing edge targets: computed on thermal positions, but the edges are the ones computed either in TH or IS
                intial_rel_positions = positions_th[edge_index[1]] - positions_th[edge_index[0]]
                initial_distances = np.sum((intial_rel_positions) ** 2, axis=-1) ** (1 / 2)
                edge_targets = []
                for timIdx in range(n_idx):
                    edge_targets.append(get_edge_targets(edge_index, initial_distances, target_positions[timIdx]))
                edge_targets = np.array(edge_targets)

                edge_index_list.append(edge_index)
                edge_attr_list.append(edge_attr)
                e_pot_list.append(e_pot)
                pair_pot_list.append(pairwise_potentials)
                edge_targets_list.append(edge_targets)


            data_out = Data(x               = torch.from_numpy(types.reshape((-1, 1))),
                            y               = torch.from_numpy(targets).T,
                            edge_index_th   = torch.from_numpy(edge_index_list[0]),
                            edge_attr_th    = torch.from_numpy(edge_attr_list[0]),
                            pos_th          = torch.from_numpy(positions_th),
                            e_pot_th        = torch.from_numpy(e_pot_list[0].reshape((-1, 1))),
                            pair_pot_th     = torch.from_numpy(pair_pot_list[0].reshape((-1, 1))),
                            edge_targets_th = torch.from_numpy(edge_targets_list[0]).T,
                            edge_index_IS   = torch.from_numpy(edge_index_list[1]),
                            edge_attr_IS    = torch.from_numpy(edge_attr_list[1]),
                            pos_IS          = torch.from_numpy(positions_IS[1]),
                            e_pot_IS        = torch.from_numpy(e_pot_list[1].reshape((-1, 1))),
                            pair_pot_IS     = torch.from_numpy(pair_pot_list[1].reshape((-1, 1))),
                            edge_targets_IS = torch.from_numpy(edge_targets_list[1]).T,
                            delta_r_cage    = torch.from_numpy((np.sum((positions_th-positions_IS) ** 2, axis=-1) ** (1 / 2)).reshape((-1, 1)))
                            )

            if self.pre_filter is not None:
                data = self.pre_filter(data)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data_out, self.processed_paths[ind])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_raw = torch.load(self.processed_paths[idx])
        x = data_raw.x
        y = data_raw.y[:,self.listTimesIdx]
        if self.IS == 1: ## WHICH RELATIVE POSITIONS AND GRAPH TO USE, is OR TH ?
            edge_index   = data_raw.edge_index_IS
            edge_attr    = data_raw.edge_attr_IS
            pos          = data_raw.pos_IS
            edge_targets = data_raw.edge_targets_IS[:,self.listTimesIdx]
        else:
            edge_index   = data_raw.edge_index_th
            edge_attr    = data_raw.edge_attr_th
            pos          = data_raw.pos_th
            edge_targets = data_raw.edge_targets_th[:,self.listTimesIdx]

        if   self.e_pot == 'IS':
            e_pot    = data_raw.e_pot_IS
            pair_pot = data_raw.pair_pot_IS
        elif self.e_pot == 'th':
            e_pot    = data_raw.e_pot_th
            pair_pot = data_raw.pair_pot_th
        else:
            e_pot    = torch.tensor(0)
            pair_pot = torch.tensor(0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos,
             e_pot=e_pot, pair_pot=pair_pot, edge_targets=edge_targets, delta_r_cage=data_raw.delta_r_cage,filename=self.processed_paths[idx])

        return data
