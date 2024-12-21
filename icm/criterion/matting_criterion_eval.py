import scipy.ndimage
import numpy as np
import torch.nn.functional as F
from skimage.measure import label
import scipy.ndimage.morphology
import torch

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(int)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_gradient_loss(pred, target, trimap):

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def compute_connectivity_error(pred, target, trimap, step):
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(int)
        flag = ((l_map == -1) & (omega == 0)).astype(int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.

def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_mse_loss_torch(pred, target, trimap):
    error_map = (pred - target) / 255.0
    # rewrite the loss with torch
    # loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)
    loss = torch.sum((error_map ** 2) * (trimap == 128).float()) / (torch.sum(trimap == 128).float() + 1e-8)

    return loss


def compute_sad_loss_torch(pred, target, trimap):
    # rewrite the map with torch
    # error_map = np.abs((pred - target) / 255.0)
    error_map = torch.abs((pred - target) / 255.0)
    # loss = np.sum(error_map * (trimap == 128))
    loss = torch.sum(error_map * (trimap == 128).float())

    return loss / 1000


def gauss_torch(x, sigma):
    """计算高斯函数值"""
    pi_tensor = torch.tensor(2 * torch.pi, device=x.device)  # 将 π 转换为张量
    y = torch.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * torch.sqrt(pi_tensor))
    return y


def dgauss_torch(x, sigma):
    """计算高斯函数的导数"""
    y = -x * gauss_torch(x, sigma) / (sigma ** 2)
    return y


def gaussgradient_torch(im, sigma):
    """计算图像的高斯梯度"""
    epsilon = torch.tensor(1e-2, device=im.device)  # 将 epsilon 转换为张量
    pi_tensor = torch.tensor(2 * torch.pi, device=im.device)  # 将 π 转换为张量
    sigma_tensor = torch.tensor(sigma, device=im.device)  # 将 sigma 转换为张量
    
    # 计算 halfsize
    halfsize = torch.ceil(sigma_tensor * torch.sqrt(-2 * torch.log(torch.sqrt(pi_tensor) * sigma_tensor * epsilon))).int()
    size = 2 * halfsize + 1
    hx = torch.zeros((size, size), device=im.device)

    for i in range(size):
        for j in range(size):
            u = torch.tensor([i - halfsize.item(), j - halfsize.item()], device=im.device)
            hx[i, j] = gauss_torch(u[0].float(), sigma) * dgauss_torch(u[1].float(), sigma)

    hx /= torch.sqrt(torch.sum(torch.abs(hx) * torch.abs(hx)))
    hy = hx.t()

    gx = F.conv2d(im.unsqueeze(0).unsqueeze(0), hx.unsqueeze(0).unsqueeze(0), padding=halfsize.item(), groups=1)
    gy = F.conv2d(im.unsqueeze(0).unsqueeze(0), hy.unsqueeze(0).unsqueeze(0), padding=halfsize.item(), groups=1)

    return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0)


def compute_gradient_loss_torch(pred, target, trimap):
    """计算预测和目标之间的梯度损失"""
    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient_torch(pred, 1.4)
    target_x, target_y = gaussgradient_torch(target, 1.4)

    pred_amp = torch.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = torch.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = torch.sum(error_map[trimap == 128])

    return loss / 1000.

def compute_connectivity_error_torch(pred, target, trimap, step):
    """计算连通性误差"""
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = torch.arange(0, 1 + step, step, device=pred.device)
    l_map = torch.ones_like(pred, dtype=torch.float32) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).int()
        target_alpha_thresh = (target >= thresh_steps[i]).int()

        omega = getLargestCC_torch(pred_alpha_thresh * target_alpha_thresh).int()
        flag = ((l_map == -1) & (omega == 0)).int()
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).int()
    target_phi = 1 - target_d * (target_d >= 0.15).int()
    loss = torch.sum(torch.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.

def getLargestCC_torch(segmentation):
    """获取最大连通组件"""
    labels = label(segmentation.cpu().numpy(), connectivity=1)  # 使用 NumPy 处理连通性
    if labels.size == 0:
        # 如果 labels 为空，返回全零的张量
        return torch.zeros_like(segmentation, dtype=torch.int, device=segmentation.device)

    # 计算每个标签的出现次数
    counts = np.bincount(labels.flat)
    if counts.size <= 1:
        # 如果没有连通组件，返回全零的张量
        return torch.zeros_like(segmentation, dtype=torch.int, device=segmentation.device)

    largest_label = np.argmax(counts[1:]) + 1  # 找到最大连通组件的标签
    largestCC = torch.tensor(labels == largest_label, device=segmentation.device)
    return largestCC