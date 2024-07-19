import os
import numpy as np
import warnings
import pickle

from torch.utils.data import Dataset , DataLoader
import torch

from torch_scatter import scatter
import torch.nn as nn
import numpy as np
import torch
import os

from PIL import Image

TRANS = -1.5

# realistic projection parameters
params = {'maxpoolz':1, 'maxpoolxy':7, 'maxpoolpadz':0, 'maxpoolpadxy':3,
            'convz':1, 'convxy':3, 'convsigmaxy':3, 'convsigmaz':1, 'convpadz':0, 'convpadxy':1,
            'imgbias':0., 'depth_bias':0.2, 'obj_ratio':0.8, 'bg_clr':0,
            'resolution':224, 'depth': 8}
warnings.filterwarnings('ignore')

class Grid2Image(nn.Module):
    """
    将3D网格转换为2D图像的PyTorch模型。
    
    使用最大池化、高斯卷积和深度通道压缩等操作来实现转换过程。

    Attributes:
        maxpool (nn.MaxPool3d): 3D最大池化层，用于密集化网格。
        conv (torch.nn.Conv3d): 3D卷积层，用于高斯平滑处理。
    """

    def __init__(self):
        """
        初始化Grid2Image模型。
        """
        super().__init__()
        # 禁用cudnn加速，确保最大池化结果一致性
        torch.backends.cudnn.benchmark = False

        # 定义最大池化层
        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']),
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'],
                                                      params['maxpoolpadxy']))
        # 定义高斯卷积层
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'], params['convpadxy'], params['convpadxy']),
                                    bias=True)
        # 获取3D高斯卷积核
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        # 将卷积核赋值给卷积层的权重
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1, 1, 1, 1, 1)
        # 将卷积层的偏置项设为0
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        """
        前向传播函数，将3D网格转换为2D图像。

        Args:
            x (torch.Tensor): 输入的3D网格，大小为 [batch, depth, resolution, resolution]。

        Returns:
            torch.Tensor: 3通道的2D图像，大小为 [batch, 3, resolution, resolution]。
        """
        # 使用最大池化层对输入3D网格进行池化，得到密集化的3D网格
        x = self.maxpool(x.unsqueeze(1))
        # 使用高斯卷积层对池化后的3D网格进行平滑处理
        x = self.conv(x)
        # 沿着z轴压缩3D网格，得到2D网格图像
        
        img = torch.max(x, dim=2)[0]
        #img = img[:, :, 2:-2, 2:-2]
    
        # 对2D网格图像进行归一化，将其值限制在[0,1]范围内
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        # 将2D网格图像反转，使背景变为前景，前景变为背景
        
        #img = torch.clip(img, 1)
        img = 1 - img
        #print(torch.min(img, dim=-1)[0])
        #print(img)
        
        #img = img*255
        ##depth img = (img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None])
        ##depth img =img - torch.min(torch.min(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        ##depth img = 255-img*255
        #print(img)
        # 在通道维度上复制3份，得到3通道的2D图像
        img = img.repeat(1, 3, 1, 1)
        #pic = img.cpu().detach().numpy()[0] #
        '''
        pic = img.detach().numpy()[0]
        pic = pic.astype(np.uint8)
        pic = Image.fromarray(pic.transpose(1, 2, 0))
        pic.save("output_image.png")

        #print(img.shape)
        #print(img.shape)
        '''
        return img

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    """将每个点云量化为3D网格。
    参数:
        points (torch.tensor): 大小为 [B, _, 3] 的点云数据
    返回:
        grid (torch.tensor): 大小为 [B * self.num_views, depth, resolution, resolution] 的网格数据
    """
    
    # 获取点云数据的批次大小、点的数量和每个点的维度（在这里维度为3）
    batch, pnum, _ = points.shape
    # 获取点云数据在每个维度上的最大值和最小值
    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    # 计算点云数据的中心点
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    # 计算点云数据在各个维度上的范围
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    # 将点云数据映射到范围[-1, 1]之间
    points = (points - pcent) / prange * 2.
    # 对点云数据的前两维进行缩放，乘以一个比例因子params['obj_ratio'](0.8)
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']
    
    # 获取深度偏置值(0.2),可以理解为将点云的 z 坐标值整体向上偏移，用于使得所有点云的 z 坐标都为正值
    depth_bias = params['depth_bias']
    # 计算点云数据在x、y、z轴上的索引，加1除2是为了将点云[-1,1]变成[0,1]，最后[0,resolution]
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1+depth_bias) * (depth - 2) #Z的范围是[0,depth-2]

    # 将x、y、z索引向上取整
    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()

    # 将x、y、z索引限制在合理范围内，避免索引越界,超出的值替换为1或resolution - 2
    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    # 计算点云数据在网格中的坐标
    coordinates = z_int * resolution * resolution + _y * resolution + _x
    # 创建一个大小为 [batch, depth, resolution, resolution] 的网格，并初始化为背景颜色params['bg_clr'] （0，0）
    grid = torch.ones([batch, depth, resolution, resolution], device=points.device).view(batch, -1) * params['bg_clr']
    
    # 将点云数据投影到网格上，使用scatter函数将点云数据按照z轴坐标投影到对应位置，并取最大值
    grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")
    # 将网格形状调整为 [batch, depth, resolution, resolution]
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0,1,3,2))

    return grid


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """
    def __init__(self):
        _views = np.asarray([
            [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            ])
        
        # adding some bias to the view angle to reveal more surface
        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            ])

        self.num_views = _views.shape[0]

        #angle = torch.tensor(_views[:, 0, :]).float().cuda() #GPU
        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        #angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        angle2 = torch.tensor(_views_bias[:, 0, :]).float()

        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        #self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)

        #self.grid2image = Grid2Image().cuda()
        self.grid2image = Grid2Image()

    def get_img(self, points):
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        grid = points2grid(points=_points, resolution=params['resolution'], depth=params['depth']).squeeze()
        img = self.grid2image(grid)
        #img_ = img.cpu()
        #pic = img_[0].detach().numpy()
        #pic = pic.astype(np.uint8)*255
        #pic = Image.fromarray(pic.transpose(1, 2, 0))
        #pic.save("output_image.png")
        return img

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param rot_mat2: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points

def get2DGaussianKernel(ksize, sigma=0):
    """
    生成一个二维高斯卷积核。
    Args:
        ksize (int): 卷积核的大小，指定了卷积核在 x 和 y 轴方向上的大小。
        sigma (float): 高斯核的标准差，默认为0，表示不进行平滑处理。
    Returns:
        kernel (torch.Tensor): 生成的二维高斯卷积核。
    """
    # 计算中心点位置
    center = ksize // 2
    # 生成坐标向量 xs，表示从中心点开始沿 x 轴方向的坐标位置
    xs = (np.arange(ksize, dtype=np.float32) - center)
    # 根据高斯函数的公式，计算一维高斯核
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    # 通过张量乘法生成二维高斯核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    # 将二维高斯核转换为 PyTorch 张量，并进行归一化，使其元素和为1
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel

def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    """
    生成一个三维高斯卷积核。
    Args:
        ksize (int): 卷积核的大小，指定了卷积核在 x 和 y 轴方向上的大小，即二维高斯核的大小。
        depth (int): 卷积核在 z 轴方向上的大小，即三维高斯核的深度。
        sigma (float): 高斯核的标准差，在 x 和 y 轴方向上的，默认为2。
        zsigma (float): 高斯核在 z 轴方向上的标准差，默认为2。
    Returns:
        kernel3d (torch.Tensor): 生成的三维高斯卷积核。
    """
    # 调用 get2DGaussianKernel 函数获取二维高斯卷积核
    kernel2d = get2DGaussianKernel(ksize, sigma)
    # 生成坐标向量 zs，表示从中心点开始沿 z 轴方向的坐标位置
    zs = (np.arange(depth, dtype=np.float32) - depth // 2)
    # 根据高斯函数的公式，计算一维高斯核 zkernel
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    # 通过广播将二维高斯核在 z 轴方向上进行复制，得到三维高斯卷积核 kernel3d
    kernel3d = np.repeat(kernel2d[None, :, :], depth, axis=0) * zkernel[:, None, None]
    # 将三维高斯卷积核进行归一化，使其元素和为1
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetFewShot(Dataset):
    def __init__(self,locate):
        self.root = '../dataset/modelnet40/ModelNetFewshot/'
        self.npoints = 8192
        self.use_normals = False
        self.num_category = 40
        self.process_data = True
        self.uniform = True
        #split = config.subset
        self.subset = locate['subset']
        self.way = locate['way']
        self.shot = locate['shot']
        self.fold = locate['fold']
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')


        print('Load processed data from %s...' % self.pickle_path)

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print('The size of data is ' + str(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, real_label = self.dataset[index]
        #print(_)
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'test':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label, real_label, index


def saveImg(img, filepath,):
    for i in range(10):
        #pic = img.cpu().detach().numpy()[i]
        pic = img.detach().numpy()[i]
        ## depth pic = pic.astype(np.uint8)
        pic = pic.astype(np.uint8)*255
        #pic = pic[:, 2:-2, 2:-2]
        pic = Image.fromarray(pic.transpose(1, 2, 0))
        pic.save(filepath + "_"+str(i)+".png")

projection = Realistic_Projection()

way = 10
shot = 20

for fold in range(0,10):
    locate = {"subset":'train',
        "way":way,
        "shot":shot,
        "fold":fold
        }
    train_set=ModelNetFewShot(locate)
    locate = {"subset":'test',
        "way":way,
        "shot":shot,
        "fold":fold
        }
    test_set =ModelNetFewShot(locate)

    for points,label,real_label,index in train_set:
        image = projection.get_img(points.unsqueeze(0))
        filepath = '../dataset/modelnet40/FewShotImg/' + str(way) + 'Way_' + str(shot) + 'Shot/' + str(fold) + '/train/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = filepath + str(index)
        #print(filepath)
        #break
        saveImg(image,filename)
    for points,label,real_label,index in test_set:
        image = projection.get_img(points.unsqueeze(0))
        filepath = '../dataset/modelnet40/FewShotImg/' + str(way) + 'Way_' + str(shot) + 'Shot/' + str(fold) + '/test/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = filepath + str(index)
        #print(filepath)
        #break
        saveImg(image,filename)
