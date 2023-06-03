import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
# import matplotlib.pyplot as plt
# from matplotlib import animation
import time

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, abs_to_relative


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


class socialGAN(object):
    def __init__(self, model_path='models/sgan-p-models/eth_8_model.pt'):
        checkpoint = torch.load(model_path)
        self.generator = get_generator(checkpoint)

    def __call__(self, np_obs_traj):
        """
        Inputs:
        - np_obs_traj: numpy of shape (seq_len, batch, 2)
        Outputs:
        - np_pred_traj: numpy of shape (seq_len, batch, 2)
        """
        with torch.no_grad():  # Inference
            # relative trajectory
            np_obs_traj_rel = abs_to_relative(np_obs_traj)
            # number of pedestrian
            num_of_pedestrian = [[0, np_obs_traj.shape[1]]]

            ### Convert from numpy to tensor
            obs_traj = torch.from_numpy(np_obs_traj).cuda()
            obs_traj_rel = torch.from_numpy(np_obs_traj_rel).cuda()
            seq_start_end = torch.tensor(num_of_pedestrian).cuda()

            # print('obs: {}, rel: {}, seq: {}'.format(obs_traj.dtype, obs_traj_rel.dtype, seq_start_end.dtype))

            ### Inference
            # start_time = time.time()
            pred_traj_rel = self.generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            # end_time = time.time()
            # print('Inference time: {}s'.format(end_time - start_time))

            ### Convert from Relative to Absolute Coordinate System
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )

            np_pred_traj = pred_traj.cpu().numpy()

        return np_pred_traj
