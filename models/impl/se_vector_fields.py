'''A sligtly modified version of the official implementation of 
"Scale equavariant CNNs with vector fields"
Paper: https://arxiv.org/pdf/1807.11783.pdf
Code: https://github.com/dmarcosg/ScaleEqNet
'''
import math
import collections
import itertools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import numpy as np
from scipy.linalg import expm, norm


def ntuple(n):
    """ Ensure that input has the correct number of elements """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, n))
    return parse


def getGrid(siz):
    """ Returns grid with coordinates from -siz[0]/2 : siz[0]/2, -siz[1]/2 : siz[1]/2, ...."""
    space = [np.linspace(-(N/2), (N/2), N) for N in siz]
    mesh = np.meshgrid(*space, indexing='ij')
    mesh = [np.expand_dims(ax.ravel(), 0) for ax in mesh]

    return np.concatenate(mesh)


def rotate_grid_2D(grid, theta):
    """ Rotate grid """
    theta = np.deg2rad(theta)

    x0 = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x1 = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)

    grid[0, :] = x0
    grid[1, :] = x1
    return grid


def rotate_grid_3D(theta, axis, grid):
    """ Rotate grid """
    theta = np.deg2rad(theta)
    axis = np.array(axis)
    rot_mat = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
    rot_mat = np.expand_dims(rot_mat, 2)
    grid = np.transpose(np.expand_dims(grid, 2), [0, 2, 1])

    return np.einsum('ijk,jik->ik', rot_mat, grid)


def get_filter_rotation_transforms(kernel_dims, angles):
    """ Return the interpolation variables needed to transform a filter by a given number of degrees """

    dim = len(kernel_dims)

    # Make grid (centered around filter-center)
    grid = getGrid(kernel_dims)

    # Rotate grid
    if dim == 2:
        grid = rotate_grid_2D(grid, angles)
    elif dim == 3:
        grid = rotate_grid_3D(angles[0], [1, 0, 0], grid)
        grid = rotate_grid_3D(angles[1], [0, 0, 1], grid)

    # Radius of filter
    radius = np.min((np.array(kernel_dims)-1) / 2.)

    # Mask out samples outside circle
    radius = np.expand_dims(radius, -1)
    dist_to_center = np.sqrt(np.sum(grid**2, axis=0))
    mask = dist_to_center >= radius+.0001
    mask = 1-mask

    # Move grid to center
    grid += radius

    return compute_interpolation_grids(grid, kernel_dims, mask)


def compute_interpolation_grids(grid, kernel_dims, mask):

    #######################################################
    # The following part is part of nd-linear interpolation

    # Add a small eps to grid so that floor and ceil operations become more stable
    grid += 0.000000001

    # Make list where each element represents a dimension
    grid = [grid[i, :] for i in range(grid.shape[0])]

    # Get left and right index (integers)
    inds_0 = [ind.astype(np.integer) for ind in grid]
    inds_1 = [ind + 1 for ind in inds_0]

    # Get weights
    weights = [float_ind - int_ind for float_ind, int_ind in zip(grid, inds_0)]

    # Special case for when ind_1 == size (while ind_0 == siz)
    # In that case we select ind_0
    ind_1_out_of_bounds = np.logical_or.reduce(
        [ind == siz for ind, siz in zip(inds_1, kernel_dims)])
    for i in range(len(inds_1)):
        inds_1[i][ind_1_out_of_bounds] = 0

    # Get samples that are out of bounds or outside mask
    inds_out_of_bounds = np.logical_or.reduce([ind < 0 for ind in itertools.chain(inds_0, inds_1)] +
                                              [ind >= siz for ind, siz in zip(inds_0, kernel_dims)] +
                                              [ind >= siz for ind, siz in zip(inds_1, kernel_dims)] +
                                              (1-mask).astype('bool')
                                              )

    # Set these samples to zero get data from upper-left-corner (which will be put to zero)
    for i in range(len(inds_0)):
        inds_0[i][inds_out_of_bounds] = 0
        inds_1[i][inds_out_of_bounds] = 0

    # Reshape
    inds_0 = [np.reshape(ind, [1, 1]+kernel_dims) for ind in inds_0]
    inds_1 = [np.reshape(ind, [1, 1]+kernel_dims) for ind in inds_1]
    weights = [np.reshape(weight, [1, 1]+kernel_dims)for weight in weights]

    # Make pytorch-tensors of the interpolation variables
    inds_0 = [Variable(torch.LongTensor(ind)) for ind in inds_0]
    inds_1 = [Variable(torch.LongTensor(ind)) for ind in inds_1]
    weights = [Variable(torch.FloatTensor(weight)) for weight in weights]

    # Make mask pytorch tensor
    mask = mask.reshape(kernel_dims)
    mask = mask.astype('float32')
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 0)
    mask = torch.FloatTensor(mask)

    # Uncomment for nearest interpolation (for debugging)
    #inds_1 = [ind*0 for ind in inds_1]
    #weights  = [weight*0 for weight in weights]

    return inds_0, inds_1, weights, mask


def apply_transform(filter, interp_vars, filters_size, old_bilinear_interpolation=True):
    """ Apply a transform specified by the interpolation_variables to a filter """

    dim = 2 if len(filter.size()) == 4 else 3

    if dim == 2:

        if old_bilinear_interpolation:
            [x0_0, x1_0], [x0_1, x1_1], [w0, w1] = interp_vars
            rotated_filter = (filter[:, :, x0_0, x1_0] * (1 - w0) * (1 - w1) +
                              filter[:, :, x0_1, x1_0] * w0 * (1 - w1) +
                              filter[:, :, x0_0, x1_1] * (1 - w0) * w1 +
                              filter[:, :, x0_1, x1_1] * w0 * w1)
        else:

            # Expand dimmentions to fit filter
            interp_vars = [[inner_el.expand_as(filter) for inner_el in outer_el]
                           for outer_el in interp_vars]

            [x0_0, x1_0], [x0_1, x1_1], [w0, w1] = interp_vars

            a = torch.gather(torch.gather(filter, 2, x0_0), 3, x1_0) * (1 - w0) * (1 - w1)
            b = torch.gather(torch.gather(filter, 2, x0_1), 3, x1_0) * w0 * (1 - w1)
            c = torch.gather(torch.gather(filter, 2, x0_0), 3, x1_1) * (1 - w0) * w1
            d = torch.gather(torch.gather(filter, 2, x0_1), 3, x1_1) * w0 * w1
            rotated_filter = a+b+c+d

        rotated_filter = rotated_filter.view(filter.size()[0], filter.size()[
                                             1], filters_size[0], filters_size[1])

    elif dim == 3:
        [x0_0, x1_0, x2_0], [x0_1, x1_1, x2_1], [w0, w1, w2] = interp_vars

        rotated_filter = (filter[x0_0, x1_0, x2_0] * (1 - w0) * (1 - w1) * (1 - w2) +
                          filter[x0_1, x1_0, x2_0] * w0 * (1 - w1) * (1 - w2) +
                          filter[x0_0, x1_1, x2_0] * (1 - w0) * w1 * (1 - w2) +
                          filter[x0_1, x1_1, x2_0] * w0 * w1 * (1 - w2) +
                          filter[x0_0, x1_0, x2_1] * (1 - w0) * (1 - w1) * w2 +
                          filter[x0_1, x1_0, x2_1] * w0 * (1 - w1) * w2 +
                          filter[x0_0, x1_1, x2_1] * (1 - w0) * w1 * w2 +
                          filter[x0_1, x1_1, x2_1] * w0 * w1 * w2)

        rotated_filter = rotated_filter.view(filter.size()[0], filter.size()[
                                             1], filters_size[0], filters_size[1], filters_size[2])

    return rotated_filter


if __name__ == '__main__':
    """ Test rotation of filter """
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.nn.parameter import Parameter
    import math
    # from sevf_utils import *

    ks = [9, 9]  # Kernel size
    angle = 45
    interp_vars = get_filter_rotation_transforms(ks, angle)

    w = Variable(torch.ones([1, 1]+ks))
    #w[:,:,4,:] = 5
    w[:, :, :, 4] = 5
    #w[:,:,0,0] = -1

    print(w)
    for angle in [0, 90, 45, 180, 65, 10]:
        print(angle, 'degrees')
        print(apply_transform(w, get_filter_rotation_transforms(ks, angle)[
              :-1], ks, old_bilinear_interpolation=True) * Variable(get_filter_rotation_transforms(ks, angle)[-1]))
        print('Difference', torch.sum(apply_transform(w, get_filter_rotation_transforms(ks, angle)[:-1], ks, old_bilinear_interpolation=False) * Variable(get_filter_rotation_transforms(
            ks, angle)[-1]) - apply_transform(w, get_filter_rotation_transforms(ks, angle)[:-1], ks, old_bilinear_interpolation=True) * Variable(get_filter_rotation_transforms(ks, angle)[-1])))


class ScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, n_scales_small=5, n_scales_big=3, mode=1, angle_range=120, output_mode=2, base=1.26):
        super(ScaleConv, self).__init__()

        kernel_size = ntuple(2)(kernel_size)
        stride = ntuple(2)(stride)
        padding = ntuple(2)(padding)
        dilation = ntuple(2)(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.n_scales_small = n_scales_small
        self.n_scales_big = n_scales_big
        self.n_scales = n_scales_small + n_scales_big
        self.angle_range = angle_range
        self.mode = mode
        self.base = base

        # Angles
        self.angles = np.linspace(-angle_range*self.n_scales_small/self.n_scales,
                                  angle_range*self.n_scales_big/self.n_scales, self.n_scales, endpoint=True)

        self.weight1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        # If input is vector field, we have two filters (one for each component)
        if self.mode == 2:
            self.weight2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.mode == 2:
            self.weight2.data.uniform_(-stdv, stdv)

    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        super(ScaleConv, self)._apply(func)

    def forward(self, input):

        if self.mode == 1:
            outputs = []
            orig_size = list(input.data.shape[2:4])
            upsample_size = (orig_size[0] // self.stride[0], orig_size[1] // self.stride[1])
            # Input upsampling scales (smaller filter scales)
            input_s = input.clone()
            for n in range(1, self.n_scales_big+1):
                size = [0, 0]
                size[0] = int(round(self.base ** n * orig_size[0]))
                size[1] = int(round(self.base ** n * orig_size[1]))
                input_s = F.upsample(input_s, size=size, mode='bilinear')
                out = F.conv2d(input_s, self.weight1, None,
                               self.stride, self.padding, self.dilation)
                # fix stride
                out = F.upsample(out, size=upsample_size, mode='bilinear')
                outputs.append(out.unsqueeze(-1))

            # Input downsampling scales (larger filter scales)
            input_s = input.clone()
            for n in range(0, self.n_scales_small):
                size = [0, 0]
                size[0] = int(round(self.base ** -n * orig_size[0]))
                size[1] = int(round(self.base ** -n * orig_size[1]))
                input_s = F.upsample(input_s, size=size, mode='bilinear')
                out = F.conv2d(input_s, self.weight1, None,
                               self.stride, self.padding, self.dilation)
                # fix stride
                out = F.upsample(out, size=upsample_size, mode='bilinear')
                outputs = [out.unsqueeze(-1)] + outputs

        if self.mode == 2:
            u = input[0]
            v = input[1]
            orig_size = list(u.data.shape[2:4])
            upsample_size = (orig_size[0] // self.stride[0], orig_size[1] // self.stride[1])
            outputs = []

            # Input upsampling scales (smaller filter scales)
            u_s = u.clone()
            v_s = v.clone()
            for n in range(1, self.n_scales_big+1):
                wu = self.weight1
                wv = self.weight2
                n_scale = self.n_scales_small + n - 1
                angle = -self.angles[n_scale] * np.pi / 180
                wru = np.cos(angle).__float__() * wu - np.sin(angle).__float__() * wv
                wrv = np.sin(angle).__float__() * wu + np.cos(angle).__float__() * wv

                size = [0, 0]
                size[0] = int(round(self.base ** n * orig_size[0]))
                size[1] = int(round(self.base ** n * orig_size[1]))
                u_s = F.upsample(u_s, size=size, mode='bilinear')
                u_out = F.conv2d(u_s, wru, None, self.stride, self.padding, self.dilation)
                u_out = F.upsample(u_out, size=upsample_size, mode='bilinear')
                v_s = F.upsample(v_s, size=size, mode='bilinear')
                v_out = F.conv2d(v_s, wrv, None, self.stride, self.padding, self.dilation)
                v_out = F.upsample(v_out, size=upsample_size, mode='bilinear')
                outputs.append((u_out + v_out).unsqueeze(-1))

            # Input downsampling scales (smaller filter scales)
            u_s = u.clone()
            v_s = v.clone()
            for n in range(0, self.n_scales_small):
                wu = self.weight1
                wv = self.weight2
                n_scale = self.n_scales_small - n - 1
                angle = -self.angles[n_scale] * np.pi / 180
                wru = np.cos(angle).__float__() * wu - np.sin(angle).__float__() * wv
                wrv = np.sin(angle).__float__() * wu + np.cos(angle).__float__() * wv

                size = [0, 0]
                size[0] = int(round(self.base ** -n * orig_size[0]))
                size[1] = int(round(self.base ** -n * orig_size[1]))
                u_s = F.upsample(u_s, size=size, mode='bilinear')
                u_out = F.conv2d(u_s, wru, None, self.stride, self.padding, self.dilation)
                u_out = F.upsample(u_out, size=upsample_size, mode='bilinear')
                v_s = F.upsample(v_s, size=size, mode='bilinear')
                v_out = F.conv2d(v_s, wrv, None, self.stride, self.padding, self.dilation)
                v_out = F.upsample(v_out, size=upsample_size, mode='bilinear')
                outputs = [(u_out + v_out).unsqueeze(-1)] + outputs

        # Get the maximum direction (Orientation Pooling)
        strength, max_ind = torch.max(torch.cat(outputs, -1), -1)

        # Convert from polar representation
        angle_map = (max_ind.float() - self.n_scales_small) * \
            np.pi/180. * self.angle_range / len(self.angles)
        u = F.relu(strength) * torch.cos(angle_map)
        v = F.relu(strength) * torch.sin(angle_map)

        return u, v


class VectorMaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False):
        super(VectorMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        # Magnitude
        p = torch.sqrt(v ** 2 + u ** 2)
        # Max pool
        _, max_inds = F.max_pool2d(p, self.kernel_size, self.stride,
                                   self.padding, self.dilation, self.ceil_mode,
                                   return_indices=True)
        # Reshape to please pytorch
        s1 = u.size()
        s2 = max_inds.size()

        max_inds = max_inds.view(s1[0], s1[1], s2[2] * s2[3])

        u = u.view(s1[0], s1[1], s1[2] * s1[3])
        v = v.view(s1[0], s1[1], s1[2] * s1[3])

        # Select u/v components according to max pool on magnitude
        u = torch.gather(u, 2, max_inds)
        v = torch.gather(v, 2, max_inds)

        # Reshape back
        u = u.view(s1[0], s1[1], s2[2], s2[3])
        v = v.view(s1[0], s1[1], s2[2], s2[3])

        return u, v


class Vector2Magnitude(nn.Module):
    def __init__(self):
        super(Vector2Magnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.sqrt(v ** 2 + u ** 2)
        return p


class Vector2Angle(nn.Module):
    def __init__(self):
        super(Vector2Angle, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.atan2(u, v)
        return p


class VectorBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):

        super(VectorBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()

    def forward(self, input):
        """
        Based on https://github.com/lberrada/bn.pytorch
        """
        if self.training:
            # Compute std
            std = self.std(input)

            alpha = self.weight / (std + self.eps)

            # update running variance
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2
            # compute output
            u = input[0] * alpha
            v = input[1] * alpha

        else:
            alpha = self.weight.data / torch.sqrt(self.running_var + self.eps)

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)
        return u, v

    def std(self, input):
        u = input[0]
        v = input[1]

        # Vector to magnitude
        p = torch.sqrt(u ** 2 + v ** 2)

        # Mean
        mu = torch.mean(p, 0, keepdim=True)
        mu = torch.mean(mu, 2, keepdim=True)
        mu = torch.mean(mu, 3, keepdim=True)

        # Variance
        var = (p) ** 2
        # This line should perharps read:
        # var = (p-mu)**2 #?

        var = torch.sum(var, 0, keepdim=True)
        var = torch.sum(var, 2, keepdim=True)
        var = torch.sum(var, 3, keepdim=True)
        std = torch.sqrt(var)

        return std


class VectorUpsampling(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(VectorUpsampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        u = F.upsample(u, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        v = F.upsample(v, size=self.size, scale_factor=self.scale_factor, mode=self.mode)

        return u, v


class VectorDropout(nn.Module):
    '''Dropout with synchronized masks
    '''

    def __init__(self, p=0.5):
        assert p < 1.0
        super().__init__()
        self.p = p

    def forward(self, input):
        u, v = input
        probs = u.data.new(u.data.size()).fill_(1-self.p)
        mask = torch.bernoulli(probs) / (1 - self.p)
        return u*mask, v*mask
