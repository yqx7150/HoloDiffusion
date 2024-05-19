import datetime
import os
from random import betavariate
import sys

import pandas as pd

sys.path.append('..')
import copy

import functools
import matplotlib.pyplot as plt
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate, io
from holo_tool import *
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error as compare_mse
# import odl
import glob
import pydicom
from cv2 import imwrite, resize
from func_test import WriteInfo
from scipy.io import loadmat, savemat
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)
from time import sleep
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, \
    mean_squared_error as mse
from cv2 import imwrite, resize

_CORRECTORS = {}
_PREDICTORS = {}


def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'


def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method  # pc
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


# ================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


# ========================================================================================================

def get_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', zl_arg=-1):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(img_model, check_num, predict, correct):
        # path = glob.glob("./Test_CT/*")

        # holoBatch = loadmat('./gt/all/gt_batch_holo.mat')['data']

        psnrAll = [0, 0, 0]
        ssimAll = [0, 0, 0]
        mseAll = [0, 0, 0]
        # ***********************************picNO
        testNUM = 100
        padding = 1200
        M, N = 512, 512
        sparse_cj = -1
        SSN = -1
        holo_result = np.zeros([testNUM, padding, padding])
        amp_result = np.zeros([testNUM, padding, padding])
        phase_result = np.zeros([testNUM, padding, padding])

        for picNO in range(0, testNUM):
            with torch.no_grad():
                # region tool
                def addNoise_king(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)
                    img_sparse[SG: SG + T, SG: SG + T] = img[SG: SG + T, SG: SG + T]
                    img_sparse[SG: SG + T, BG: BG + T] = img[SG: SG + T, BG: BG + T]
                    img_sparse[BG: BG + T, SG: SG + T] = img[BG: BG + T, SG: SG + T]
                    img_sparse[BG: BG + T, BG: BG + T] = img[BG: BG + T, BG: BG + T]
                    return img_sparse

                def DC_king(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img[SG: SG + T, SG: SG + T] = gtImg[SG: SG + T, SG: SG + T]
                    img[SG: SG + T, BG: BG + T] = gtImg[SG: SG + T, BG: BG + T]
                    img[BG: BG + T, SG: SG + T] = gtImg[BG: BG + T, SG: SG + T]
                    img[BG: BG + T, BG: BG + T] = gtImg[BG: BG + T, BG: BG + T]

                    return img

                # 稀疏角与resize，只支持R2R3R4
                def addNoise_sparse(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)

                    block1 = img[SG: SG + T: sparse_cj, SG: SG + T]
                    block1 = resize(block1, (T, T))
                    block2 = img[SG: SG + T: sparse_cj, BG: BG + T]
                    block2 = resize(block2, (T, T))
                    block3 = img[BG: BG + T: sparse_cj, SG: SG + T]
                    block3 = resize(block3, (T, T))
                    block4 = img[BG: BG + T: sparse_cj, BG: BG + T]
                    block4 = resize(block4, (T, T))

                    img_sparse[SG: SG + T, SG: SG + T] = block1
                    img_sparse[SG: SG + T, BG: BG + T] = block2
                    img_sparse[BG: BG + T, SG: SG + T] = block3
                    img_sparse[BG: BG + T, BG: BG + T] = block4

                    return img_sparse

                def DC_sparse(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    # img[SG: SG + T : sparse_cj, SG: SG + T] = gtImg[SG: SG + T: sparse_cj, SG: SG + T]
                    # img[SG: SG + T: sparse_cj, BG: BG + T] = gtImg[SG: SG + T: sparse_cj, BG: BG + T]
                    # img[BG: BG + T: sparse_cj, SG: SG + T] = gtImg[BG: BG + T: sparse_cj, SG: SG + T]
                    # img[BG: BG + T: sparse_cj, BG: BG + T] = gtImg[BG: BG + T: sparse_cj, BG: BG + T]

                    block1 = gtImg[SG: SG + T: sparse_cj, SG: SG + T]
                    block1 = resize(block1, (T, T))
                    block2 = gtImg[SG: SG + T: sparse_cj, BG: BG + T]
                    block2 = resize(block2, (T, T))
                    block3 = gtImg[BG: BG + T: sparse_cj, SG: SG + T]
                    block3 = resize(block3, (T, T))
                    block4 = gtImg[BG: BG + T: sparse_cj, BG: BG + T]
                    block4 = resize(block4, (T, T))

                    img[SG: SG + T, SG: SG + T] = block1
                    img[SG: SG + T, BG: BG + T] = block2
                    img[BG: BG + T, SG: SG + T] = block3
                    img[BG: BG + T, BG: BG + T] = block4

                    return img

                # 补丁：增加了2/3,3/4的选项，sparse_cj=3就是采集2/3
                def addNoise_sparse_mix(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)
                    #img_sparse = np.zeros(img.shape, np.float32)

                    timeToSkip = 0
                    for i in range(0, size):
                        # 一组一组采集
                        if timeToSkip % sparse_cj == 0:
                            addTemp = i + sparse_cj - 1 #  -1就是少采的那一行
                            img_sparse[SG + i: SG + addTemp, SG: SG + T] = img[SG + i: SG + addTemp, SG: SG + T]
                            img_sparse[SG + i: SG + addTemp, BG: BG + T] = img[SG + i: SG + addTemp, BG: BG + T]
                            img_sparse[BG + i: BG + addTemp, SG: SG + T] = img[BG + i: BG + addTemp, SG: SG + T]
                            img_sparse[BG + i: BG + addTemp, BG: BG + T] = img[BG + i: BG + addTemp, BG: BG + T]
                        timeToSkip = timeToSkip + 1


                    return img_sparse

                def DC_sparse_mix(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap
                    timeToSkip = 0
                    for i in range(0, size):
                        timeToSkip = timeToSkip + 1
                        if timeToSkip % sparse_cj == 0:
                            addTemp = i + sparse_cj - 1 #- 1
                            img[SG + i: SG + addTemp, SG: SG + T] = gtImg[SG + i: SG + addTemp, SG: SG + T]
                            img[SG + i: SG + addTemp, BG: BG + T] = gtImg[SG + i: SG + addTemp, BG: BG + T]
                            img[BG + i: BG + addTemp, SG: SG + T] = gtImg[BG + i: BG + addTemp, SG: SG + T]
                            img[BG + i: BG + addTemp, BG: BG + T] = gtImg[BG + i: BG + addTemp, BG: BG + T]

                    return img

                # 行列都欠采，然后resize
                def addNoise_sparse_SR(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)

                    block1 = img[SG: SG + T, SG: SG + T]
                    #savemat('abcd_ori.mat', mdict={'data': block1})
                    block1 = cutRowColunm(block1,size)
                    #savemat('abcd_cut.mat', mdict={'data': block1})
                    block1 = resize(block1, (T, T))
                    #savemat('abcd_resize.mat', mdict={'data': block1})


                    block2 = img[SG: SG + T, BG: BG + T]
                    block2 = cutRowColunm(block2,size)
                    block2 = resize(block2, (T, T))

                    block3 = img[BG: BG + T, SG: SG + T]
                    block3 = cutRowColunm(block3,size)
                    block3 = resize(block3, (T, T))

                    block4 = img[BG: BG + T, BG: BG + T]
                    block4 = cutRowColunm(block4, size)
                    block4 = resize(block4, (T, T))

                    img_sparse[SG: SG + T, SG: SG + T] = block1
                    img_sparse[SG: SG + T, BG: BG + T] = block2
                    img_sparse[BG: BG + T, SG: SG + T] = block3
                    img_sparse[BG: BG + T, BG: BG + T] = block4

                    return img_sparse

                def DC_sparse_SR(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    # img[SG: SG + T : sparse_cj, SG: SG + T] = gtImg[SG: SG + T: sparse_cj, SG: SG + T]
                    # img[SG: SG + T: sparse_cj, BG: BG + T] = gtImg[SG: SG + T: sparse_cj, BG: BG + T]
                    # img[BG: BG + T: sparse_cj, SG: SG + T] = gtImg[BG: BG + T: sparse_cj, SG: SG + T]
                    # img[BG: BG + T: sparse_cj, BG: BG + T] = gtImg[BG: BG + T: sparse_cj, BG: BG + T]

                    block1 = gtImg[SG: SG + T, SG: SG + T]
                    block1 = cutRowColunm(block1, size)
                    block1 = resize(block1, (T, T))

                    block2 = gtImg[SG: SG + T, BG: BG + T]
                    block2 = cutRowColunm(block2, size)
                    block2 = resize(block2, (T, T))

                    block3 = gtImg[BG: BG + T, SG: SG + T]
                    block3 = cutRowColunm(block3, size)
                    block3 = resize(block3, (T, T))

                    block4 = gtImg[BG: BG + T, BG: BG + T]
                    block4 = cutRowColunm(block4, size)
                    block4 = resize(block4, (T, T))

                    img[SG: SG + T, SG: SG + T] = block1
                    img[SG: SG + T, BG: BG + T] = block2
                    img[BG: BG + T, SG: SG + T] = block3
                    img[BG: BG + T, BG: BG + T] = block4

                    return img

                def cutRowColunm(block,size):
                    fz = fzfm[0]
                    fm = fzfm[1]

                    # 删除列：
                    arr = np.array([], dtype=int)

                    for i in range(0, size // fm):
                        needToAdd = np.array([], dtype=int)
                        for j in range(fz):
                            needToAdd = np.append(needToAdd, [i * fm + fm - fz + j])
                        arr = np.append(arr, [needToAdd])

                    # block = np.delete(block, arr, axis=0)
                    # block = np.delete(block, arr, axis=1)

                    block = np.delete(block, arr, axis=0)
                    block = np.delete(block, arr, axis=1)



                    return block

                # 行列都欠采，不resize
                def addNoise_sparse_SR_2(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)

                    gt = img[SG: SG + T, SG: SG + T]
                    block = img_sparse[SG: SG + T, SG: SG + T]
                    block1 = small_DC(block, gt, size)

                    gt = img[SG: SG + T, BG: BG + T]
                    block = img_sparse[SG: SG + T, BG: BG + T]
                    block2 = small_DC(block, gt, size)

                    gt = img[BG: BG + T, SG: SG + T]
                    block = img_sparse[BG: BG + T, SG: SG + T]
                    block3 = small_DC(block, gt, size)

                    gt = img[BG: BG + T, BG: BG + T]
                    block = img_sparse[BG: BG + T, BG: BG + T]
                    block4 = small_DC(block, gt, size)

                    img_sparse[SG: SG + T, SG: SG + T] = block1
                    img_sparse[SG: SG + T, BG: BG + T] = block2
                    img_sparse[BG: BG + T, SG: SG + T] = block3
                    img_sparse[BG: BG + T, BG: BG + T] = block4

                    return img_sparse

                def DC_sparse_SR_2(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    gt = gtImg[SG: SG + T, SG: SG + T]
                    block = img[SG: SG + T, SG: SG + T]
                    block1 = small_DC(block, gt, size)

                    gt = gtImg[SG: SG + T, BG: BG + T]
                    block = img[SG: SG + T, BG: BG + T]
                    block2 = small_DC(block, gt, size)

                    gt = gtImg[BG: BG + T, SG: SG + T]
                    block = img[BG: BG + T, SG: SG + T]
                    block3 = small_DC(block, gt, size)

                    gt = gtImg[BG: BG + T, BG: BG + T]
                    block = img[BG: BG + T, BG: BG + T]
                    block4 = small_DC(block, gt, size)


                    img[SG: SG + T, SG: SG + T] = block1
                    img[SG: SG + T, BG: BG + T] = block2
                    img[BG: BG + T, SG: SG + T] = block3
                    img[BG: BG + T, BG: BG + T] = block4

                    return img

                def small_DC(block, gt, size):
                    road = np.ones(block.shape, dtype=int)

                    fz = fzfm[0]
                    fm = fzfm[1]
                    # 删除列：
                    arr = np.array([], dtype=int)
                    for i in range(0, size // fm):
                        needToAdd = np.array([], dtype=int)
                        for j in range(fz):
                            needToAdd = np.append(needToAdd, [i * fm + fm - fz + j])
                        arr = np.append(arr, [needToAdd])

                    road[arr, :] = -666
                    road[:, arr] = -666
                    road[road != -666] = 1

                    for i in range(road.shape[0]):
                        for j in range(road.shape[1]):
                            if road[i, j] == 1:
                                block[i, j] = gt[i, j]
                    return block


                # SSN
                def addNoise_sensor(img, size=200, picSize=512, gap=10):
                    if gap + size >= picSize / 2:
                        raise Exception("hey man, your gap + size is bigger than picSize!")
                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    img_sparse = np.ones(img.shape, np.float32)
                    JZ = (picSize-size)//2
                    if SSN == 1:
                        img_sparse[SG: SG + T, SG: SG + T] = img[SG: SG + T, SG: SG + T]  # 1
                    elif SSN == 2:
                        img_sparse[SG: SG + T, SG: SG + T] = img[SG: SG + T, SG: SG + T] # 1
                        img_sparse[BG: BG + T, BG: BG + T] = img[BG: BG + T, BG: BG + T] # 4
                    elif SSN == 3:
                        img_sparse[SG: SG + T, JZ: JZ + T] = img[SG: SG + T, JZ: JZ + T] # mid 1
                        img_sparse[BG: BG + T, SG: SG + T] = img[BG: BG + T, SG: SG + T] # 3
                        img_sparse[BG: BG + T, BG: BG + T] = img[BG: BG + T, BG: BG + T] # 4
                    elif SSN == 4:
                        img_sparse[SG: SG + T, SG: SG + T] = img[SG: SG + T, SG: SG + T] # 1
                        img_sparse[SG: SG + T, BG: BG + T] = img[SG: SG + T, BG: BG + T] # 2
                        img_sparse[BG: BG + T, SG: SG + T] = img[BG: BG + T, SG: SG + T] # 3
                        img_sparse[BG: BG + T, BG: BG + T] = img[BG: BG + T, BG: BG + T] # 4

                    return img_sparse

                def DC_sensor(img, gtImg, size=200, picSize=512, gap=10):
                    if gap + size >= picSize:
                        raise Exception("hey man, your gap + size is bigger than picSize!")

                    SG = int(picSize / 2 - gap - size)
                    T = size
                    BG = size + 2 * gap + SG  # picSize // 2 + gap

                    JZ = (picSize-size)//2
                    if SSN == 1:
                        img[SG: SG + T, SG: SG + T] = gtImg[SG: SG + T, SG: SG + T]  # 1
                    elif SSN == 2:
                        img[SG: SG + T, SG: SG + T] = gtImg[SG: SG + T, SG: SG + T] # 1
                        img[BG: BG + T, BG: BG + T] = gtImg[BG: BG + T, BG: BG + T] # 4
                    elif SSN == 3:
                        img[SG: SG + T, JZ: JZ + T] = gtImg[SG: SG + T, JZ: JZ + T] # mid 1
                        img[BG: BG + T, SG: SG + T] = gtImg[BG: BG + T, SG: SG + T] # 3
                        img[BG: BG + T, BG: BG + T] = gtImg[BG: BG + T, BG: BG + T] # 4
                    elif SSN == 4:
                        img[SG: SG + T, SG: SG + T] = gtImg[SG: SG + T, SG: SG + T] # 1
                        img[SG: SG + T, BG: BG + T] = gtImg[SG: SG + T, BG: BG + T] # 2
                        img[BG: BG + T, SG: SG + T] = gtImg[BG: BG + T, SG: SG + T] # 3
                        img[BG: BG + T, BG: BG + T] = gtImg[BG: BG + T, BG: BG + T] # 4

                    return img

                def holoSplit(holo, ifAbs=True):
                    if ifAbs:
                        holo = abs(holo)
                    complex_img = IFT((FT(holo)) * np.conj(prop))
                    amp_r = abs(complex_img)
                    phase_r = np.angle(complex_img)
                    return amp_r, phase_r

                def toNumpy(tensor):
                    return np.squeeze(tensor.cpu().numpy())

                def countPSM(aa, bb):
                    aa = np.squeeze(aa)
                    bb = np.squeeze(bb)

                    # 归一化
                    maxvalue1 = np.max(aa)
                    minvalue1 = np.min(aa)
                    aa = (aa - minvalue1) / (maxvalue1 - minvalue1)
                    maxvalue1 = np.max(bb)
                    minvalue1 = np.min(bb)
                    bb = (bb - minvalue1) / (maxvalue1 - minvalue1)

                    # x0 = aa
                    # print(f"the aa's max is : {np.max(x0)}, min is : {np.min(x0)} shape is :{x0.shape}")
                    # plt.imshow(x0, cmap=plt.get_cmap('gray'))
                    # plt.show()
                    #
                    # x0 = bb
                    # print(f"the bb's max is : {np.max(x0)}, min is : {np.min(x0)} shape is :{x0.shape}")
                    # plt.imshow(x0, cmap=plt.get_cmap('gray'))
                    # plt.show()

                    psnr0 = psnr(aa, bb, data_range=1)
                    ssim0 = ssim(aa, bb, gaussian_weights=True, use_sample_covariance=False, data_range=1.0)
                    ssim0 = ssim(aa, bb, data_range=1.0)
                    mse0 = mse(aa, bb)
                    psnr0 = round(psnr0, 2)
                    ssim0 = round(ssim0, 4)
                    mse0 = round(mse0, 6)

                    return psnr0, ssim0, mse0

                def WriteInfo(path, **args):
                    """
                    ### 写入结果至CSV文件
                    ###   path : 文件路径
                    ### **args : 需写入的变量数据,同时以标量或列表形式传入:
                        write_info('./raki_result.csv',psnr =[32.2],mse = [1.54],ssim= [0.9756],mae=[0.12])
                    """
                    ppp = os.path.split(path)
                    if not os.path.isdir(ppp[0]):
                        os.makedirs(ppp[0])
                        # print(f"{pathDir} 创建成功")

                    try:
                        args = args['args']
                    except:
                        pass
                    # print(args)
                    # assert 0
                    args['Time'] = [str(datetime.datetime.now())[:-7]]
                    try:
                        df = pd.read_csv(path, encoding='utf-8', engine='python')
                    except:
                        df = pd.DataFrame()
                    df2 = pd.DataFrame(args)
                    df = df.append(df2)
                    df.to_csv(path, index=False)

                def savePng(path, img):
                    ppp = os.path.split(path)
                    if not os.path.isdir(ppp[0]):
                        os.makedirs(ppp[0])
                        # print(f"{pathDir} 创建成功")

                    plt.imshow(img, cmap=plt.get_cmap('gray'))
                    plt.savefig(f'{path}')

                def getInsideIndex(pad=padding, size=N):
                    return int((pad - size) / 2)

                def addPaddingAmp(amp, pad=padding, size=N):
                    # amp is 512*512
                    temp = np.ones((pad, pad), np.float32)

                    if size > 512:
                        raise Exception("hey man, your cut is too big!")

                    inIn = getInsideIndex(512, size)
                    amp_cut = amp[inIn:inIn + size, inIn:inIn + size]

                    inIn = getInsideIndex(pad, size)
                    temp[inIn:inIn + size, inIn:inIn + size] = amp_cut
                    return temp

                def addPaddingPhase(phase, pad=padding, size=N):
                    temp = np.zeros((pad, pad), np.float32)

                    if size > 512:
                        raise Exception("hey man, your cut is too big!")

                    inIn = getInsideIndex(512, size)
                    phase_cut = phase[inIn:inIn + size, inIn:inIn + size]

                    inIn = getInsideIndex(pad, size)
                    temp[inIn:inIn + size, inIn:inIn + size] = phase_cut
                    return temp

                def getFormatPSM(psnr2, ssim2, mse2):
                    return f"{('%.2f' % psnr2)}/{('%.4f' % ssim2)}/{('%.4f' % mse2)}"
                # endregion
                # ---------------------------------------------------

                learning_Objet = 4
                #                 0    1     2       3          4
                learning_List = ['0', '7', 'all', 'letter', 'all_100']
                learning_Objet = learning_List[learning_Objet]
                ampBatch = loadmat(f'./gt/{learning_Objet}/gt_batch_amp.mat')['data']
                phaseBatch = loadmat(f'./gt/{learning_Objet}/gt_batch_phase.mat')['data']
                print(f"================== Now we testing the {learning_Objet} ==================")

                # savePath = 'result_m20_L7TL/SSN_Special_Ones'
                savePath = 'result_m20/L7TA_100_sparse_3'
                # savePath = 'result_m20_allNumber/SR_sparse_resize_test'

                useSrSparse = False
                useSparse = False
                useSSN = False
                fzfm=[1,5]

                sparse_cj = zl_arg.sparse
                SSN = zl_arg.ssn  # auto
                if SSN > 0:
                    useSSN=True
                if sparse_cj > 0:
                    useSparse = True


                ifSavePng = False
                ifSaveAll = True
                showImg = False # !!!!!! must close!!

                startStep = 1600
                endStep = 2100  # 2100

                keep_size = zl_arg.size  # 400
                useNet = zl_arg.useNet  # True
                gap = zl_arg.gap  # auto

                maskType = 'default_0'
                if useSrSparse:
                    addNoise = addNoise_sparse_SR_2
                    DC = DC_sparse_SR_2
                    maskType = f'fzfm_{fzfm}'
                    print(f"================== Now we use fzfm {fzfm}==================")
                elif useSparse:
                    # addNoise = addNoise_sparse
                    # DC = DC_sparse
                    addNoise = addNoise_sparse_mix
                    DC = DC_sparse_mix
                    maskType = f'sparse_{sparse_cj}'
                    print(f"================== Now we use sparse {sparse_cj} ==================")
                elif useSSN:
                    addNoise = addNoise_sensor
                    DC = DC_sensor
                    maskType = f'SSN_{SSN}'
                    print(f"================== Now we use SSN {SSN} ==================")
                else:
                    addNoise = addNoise_king
                    DC = DC_king
                    print(f"================== Now we use nothing noise just size and gap ==================")
                inIn = getInsideIndex(padding, N)

                wavelength = 500 * (pow(10, -9))
                range1 = 0.001 / (1200 / padding)
                z = 0.0024
                prop = Propagator_function(padding, z, wavelength, range1)

                amp_gt = ampBatch[picNO, :, :]
                phase_gt = phaseBatch[picNO, :, :]
                amp_gt_b = addPaddingAmp(amp_gt)
                phase_gt_b = addPaddingPhase(phase_gt)

                fs_b = amp_gt_b * np.exp(1j * phase_gt_b)
                U = IFT((FT(fs_b)) * prop)
                holo_gt_b = abs(U)
                # region else
                title = 'NCSNPP' if useNet else 'SRSAA'  # SRSAA NCSNPP
                title = f'{title}_{maskType}'
                # 保存原图
                savePng(f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/phase/gt.png',
                        phase_gt_b)
                savePng(f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/amp/gt.png',
                        amp_gt_b)
                savePng(f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/holo/gt.png',
                        abs(holo_gt_b))
                # endregion
                holo_sparse_b = addNoise(holo_gt_b, size=keep_size, picSize=padding, gap=gap)

                # mat
                #savemat('aaaa.mat', mdict={'data': abs(holo_sparse_b)})

                savePng(f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/holo/g_sparse.png',
                        abs(holo_sparse_b))

                timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
                support0 = supportPro(padding)

                phase_b = np.zeros((padding, padding), np.float32)
                # phase_b = phase_gt_b

                psnrMax = [1, 1, 1]
                ssimMax = [0, 0, 0]
                mseMax = [99, 99, 99]

                bestHolo = np.zeros((padding, padding), np.float32)
                bestAmp = np.zeros((padding, padding), np.float32)
                bestPhase = np.zeros((padding, padding), np.float32)

                for i in range(startStep, endStep):
                    t = timesteps[startStep]
                    step = i - startStep
                    vec_t = torch.ones(1, device=t.device) * t

                    # holo_sparse_b[inIn:inIn+N, inIn:inIn+N] = DC(holo_sparse_b[inIn:inIn+N, inIn:inIn+N], holo_gt, size=keep_size,picSize=padding)
                    holo_sparse_b = DC(holo_sparse_b, holo_gt_b, size=keep_size, picSize=padding, gap=gap)

                    # region SRSAA_code
                    fushutu = holo_sparse_b * np.exp(1j * phase_b)
                    t1 = IFT((FT(fushutu)) * np.conj(prop))

                    object_b = 1 - abs(t1)
                    phase_b2 = np.angle(t1)

                    for ii in range(padding):
                        for jj in range(padding):
                            if object_b[ii, jj] < 0:
                                object_b[ii, jj] = 0
                                phase_b2[ii, jj] = 0

                            if phase_b2[ii, jj] <= -1:
                                phase_b2[ii, jj] = -1
                            if phase_b2[ii, jj] >= 1:
                                phase_b2[ii, jj] = 1


                    object_b = object_b * support0
                    object_b = 1 - object_b

                    # **************************** NET START ***********************
                    amp_phase = np.concatenate(
                        [object_b[None, inIn:inIn + N, inIn:inIn + N], phase_b2[None, inIn:inIn + N, inIn:inIn + N]],
                        axis=0)
                    amp_phase = np.float32(amp_phase)
                    amp_phase = torch.from_numpy(amp_phase[None, :, :, :]).cuda()
                    if useNet:
                        if showImg and i % 5 == 0:
                            amp_phase_temp = toNumpy(amp_phase)
                            plt.imshow(amp_phase_temp[1], cmap=plt.get_cmap('gray'))
                            plt.show()

                        amp_phase1, amp_phase = predictor_update_fn(amp_phase, vec_t, model=img_model)

                        if showImg and i % 5 == 0:
                            amp_phase_temp = toNumpy(amp_phase)
                            plt.imshow(amp_phase_temp[1], cmap=plt.get_cmap('gray'))
                            plt.show()

                        # amp_phase1, amp_phase = corrector_update_fn(amp_phase, vec_t, model=img_model)

                        if showImg and i % 5 == 0:
                            amp_phase_temp = toNumpy(amp_phase)
                            plt.imshow(amp_phase_temp[1], cmap=plt.get_cmap('gray'))
                            plt.show()

                    amp_phase = toNumpy(amp_phase)
                    # **************************  NET END ***********************
                    object_b[inIn:inIn + N, inIn:inIn + N] = amp_phase[0]
                    phase_b2[inIn:inIn + N, inIn:inIn + N] = amp_phase[1]

                    # add more DC here
                    # object_b = 1 - abs(object_b)
                    # for ii in range(padding):
                    #     for jj in range(padding):
                    #         if object_b[ii, jj] < 0:
                    #             object_b[ii, jj] = 0
                    #             phase_b2[ii, jj] = 0
                    # object_b = object_b * support0
                    # object_b = 1 - object_b

                    t1 = object_b * np.exp(1j * phase_b2)
                    holo_field_updated = IFT((FT(t1)) * prop)
                    holo_sparse_b = abs(holo_field_updated)
                    phase_b = np.angle(holo_field_updated)

                    if showImg and i % 5 == 0:
                        plt.imshow(np.angle(t1), cmap=plt.get_cmap('gray'))
                        plt.show()

                    # endregion

                    if i % 1 == 0:
                        img_rec_pow = abs(holo_field_updated) ** 2
                        img_abs_pow = holo_gt_b ** 2

                        psnr_holo, ssim_holo, mse_holo = countPSM(img_abs_pow, img_rec_pow)

                        # amp_gt, phase_gt = holoSplit(img_abs_pow)
                        amp_r, phase_r = holoSplit(img_rec_pow)

                        cutStart = 585  # 241 512/2 - 14 - 1
                        cutSize = 28  # 28 N - cutStart*2 - 2

                        phase_gt_small = phase_gt_b[cutStart:cutStart + cutSize, cutStart:cutStart + cutSize]
                        phase_r_small = phase_r[cutStart:cutStart + cutSize, cutStart:cutStart + cutSize]

                        amp_gt_small = amp_gt_b[cutStart:cutStart + cutSize, cutStart:cutStart + cutSize]
                        amp_r_small = amp_r[cutStart:cutStart + cutSize, cutStart:cutStart + cutSize]

                        psnr_amp, ssim_amp, mse_amp = countPSM(amp_r_small, amp_gt_small)
                        psnr_phase, ssim_phase, mse_phase = countPSM(phase_r_small, phase_gt_small)
                        if i % 5 == 0:
                            print(
                                f"size {keep_size} gap {gap} Net {useNet} sparse {useSparse} picNo {picNO} holo {step}  psnr {'%.4f' % psnr_holo}  ssim {'%.4f' % ssim_holo}  mse {'%.4f' % mse_holo}")
                            print(
                                f"size {keep_size} gap {gap} Net {useNet} sparse {useSparse} amp {step}                                                   psnr {'%.4f' % psnr_amp}  ssim {'%.4f' % ssim_amp}  mse {'%.4f' % mse_amp} ")
                            print(
                                f"size {keep_size} gap {gap} Net {useNet} sparse {useSparse} phase {step}                                                                                          psnr {'%.4f' % psnr_phase} ssim {'%.4f' % ssim_phase}  mse {'%.4f' % mse_phase}")

                        # if ssimMax[0] < ssim_holo:
                        if mseMax[2] > mse_phase:
                            # print("yes, yes, we will update it ")
                            psnrMax[0] = psnr_holo
                            ssimMax[0] = ssim_holo
                            mseMax[0] = mse_holo

                            psnrMax[1] = psnr_amp
                            ssimMax[1] = ssim_amp
                            mseMax[1] = mse_amp

                            psnrMax[2] = psnr_phase
                            ssimMax[2] = ssim_phase
                            mseMax[2] = mse_phase

                            bestHolo = img_rec_pow
                            bestAmp = amp_r
                            bestPhase = phase_r

                        if i == endStep - 1:
                            WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}/all_ave.csv',
                                      picNO=picNO, type='holo',
                                      PSM=getFormatPSM(psnrMax[0], ssimMax[0], mseMax[0]), Check_num='-----')
                            WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}/all_ave.csv',
                                      picNO=picNO, type='phase',
                                      PSM=getFormatPSM(psnrMax[2], ssimMax[2], mseMax[2]), Check_num='-----')
                            WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}/all_ave.csv',
                                      picNO=picNO, type='amp',
                                      PSM=getFormatPSM(psnrMax[1], ssimMax[1], mseMax[1]), Check_num='-----')


                        # WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}_holo/holo_all.csv',
                        #           picNO=picNO, step=step, PSNR=psnr_holo, SSIM=ssim_holo, MSE=mse_holo,
                        #           Check_num=check_num)
                        #
                        # WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}_amp/amp_all.csv',
                        #           picNO=picNO, step=step, PSNR=psnr_amp, SSIM=ssim_amp, MSE=mse_amp,
                        #           Check_num=check_num)
                        #
                        # WriteInfo(f'./{savePath}/else/{title}_size_{keep_size}_gap_{gap}_phase/phase_all.csv',
                        #           picNO=picNO, step=step, PSNR=psnr_phase, SSIM=ssim_phase, MSE=mse_phase,
                        #           Check_num=check_num)


                        if ifSavePng and i % 50 == 0:
                            if ifSaveAll:
                                savePng(
                                    f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/holo/{step}.png',
                                    img_rec_pow)
                                savePng(
                                    f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/amp/{step}.png',
                                    amp_r)
                            savePng(
                                f'./{savePath}/pic/{title}_size_{keep_size}_gap_{gap}_iter/{picNO}/phase/{step}.png',
                                phase_r)

                            # savemat(f'./{savePath}/pic/{title}_phase_size_{keep_size}_gap_{gap}_iter/{picNO}/holo/{step}.mat', mdict={'data': phase_r})

                # region countPSM
                psnrAll[0] = psnrAll[0] + psnrMax[0]
                ssimAll[0] = ssimAll[0] + ssimMax[0]
                mseAll[0] = mseAll[0] + mseMax[0]

                psnrAll[1] = psnrAll[1] + psnrMax[1]
                ssimAll[1] = ssimAll[1] + ssimMax[1]
                mseAll[1] = mseAll[1] + mseMax[1]

                psnrAll[2] = psnrAll[2] + psnrMax[2]
                ssimAll[2] = ssimAll[2] + ssimMax[2]
                mseAll[2] = mseAll[2] + mseMax[2]
                # endregion

                holo_result[picNO, ...] = bestHolo
                amp_result[picNO, ...] = bestAmp
                phase_result[picNO, ...] = bestPhase

        WriteInfo(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/all_ave.csv', type='holo',
                  PSM=getFormatPSM(psnrAll[0] / testNUM, ssimAll[0] / testNUM, mseAll[0] / testNUM),
                  Check_num='-----')

        WriteInfo(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/all_ave.csv', type='phase',
                  PSM=getFormatPSM(psnrAll[2] / testNUM, ssimAll[2] / testNUM, mseAll[2] / testNUM),
                  Check_num='-----')

        WriteInfo(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/all_ave.csv', type='amp',
                  PSM=getFormatPSM(psnrAll[1] / testNUM, ssimAll[1] / testNUM, mseAll[1] / testNUM),
                  Check_num='-----')

        # np.save(f'./result_{picNumber}/noise_{self.noise}/chch_img.npy',holo_result)
        if not os.path.isdir(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch'):
            os.makedirs(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch')

        io.savemat(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/CN_{check_num}_holo_batch.mat',
                   {"data": holo_result})
        io.savemat(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/CN_{check_num}_amp_batch.mat',
                   {"data": amp_result})
        io.savemat(f'./{savePath}/batch/{title}_size_{keep_size}_gap_{gap}_batch/CN_{check_num}_phase_batch.mat',
                   {"data": phase_result})

    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
