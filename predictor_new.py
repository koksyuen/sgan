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
from sgan.utils import relative_to_abs, abs_to_relative, torch_abs_to_relative


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
    # generator.train()
    generator.eval()

    for param in generator.parameters():
        param.requires_grad = False

    return generator


class socialGAN(object):
    def __init__(self, model_path='../sgan/models/sgan-p-models/eth_8_model.pt'):
        checkpoint = torch.load(model_path)
        self.generator = get_generator(checkpoint)

    def __call__(self, obs_traj, seq_start_end):
        """
        Inputs:
        - np_obs_traj: numpy of shape (seq_len, batch, 2)
        Outputs:
        - np_pred_traj: numpy of shape (seq_len, batch, 2)
        """
        with torch.no_grad():  # Inference

            ### Convert from numpy to tensor
            obs_traj_rel = torch_abs_to_relative(obs_traj)

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

        return pred_traj
