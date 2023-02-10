def anorm(p1,p2,p11,p21): #p1[0] x  p1[1] y
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    NORM1 = ((p11[0]-p1[0])/(p11[1]-p1[1]))*((p21[0]-p2[0])/(p21[1]-p2[1]))
    if NORM1 < 0:
        return 0
    return 1/(NORM)

V = np.zeros((seq_len, max_nodes, 2))
A = np.zeros((seq_len, max_nodes, max_nodes))
for s in range(seq_len-1):  # s=8
    step_ = seq_[:, :, s]  # 取对应的那一帧的信息
    step_rel = seq_rel[:, :, s]
    step_1 = seq_[:, :, s+1]  # 取对应的那一帧的信息
    step_rel1 = seq_rel[:, :, s+1]

    for h in range(len(step_)):  # h=max_nodes
        V[s, h, :] = step_rel[h]  # 取对应帧的对应行人的信息
        V[s+1, h, :] = step_rel1[h]
        A[s, h, h] = 1
        A[s+1, h, h] = 1
        for k in range(h + 1, len(step_)):  # len(step_)=max_nodes
            l2_norm = anorm(step_rel[h], step_rel[k],step_rel1[h], step_rel1[k])  # V[s,h,h],V[s,h,h+1]
            A[s, h, k] = l2_norm
            A[s, k, h] = l2_norm
    if norm_lap_matr:
        G = nx.from_numpy_matrix(A[s, :, :])
        A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

import torch
import torch.nn as nn
import torchvision


class GlobalContextBlock(nn.Module):
            def __init__(self,
                         inplanes,
                         ratio,
                         pooling_type='att',
                         fusion_types=('channel_add',)):
                super(GlobalContextBlock, self).__init__()
                assert pooling_type in ['avg', 'att']
                assert isinstance(fusion_types, (list, tuple))
                valid_fusion_types = ['channel_add', 'channel_mul']
                assert all([f in valid_fusion_types for f in fusion_types])
                assert len(fusion_types) > 0, 'at least one fusion should be used'
                self.inplanes = inplanes
                self.ratio = ratio
                self.planes = int(inplanes * ratio)
                self.pooling_type = pooling_type
                self.fusion_types = fusion_types
                if pooling_type == 'att':
                    self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
                    self.softmax = nn.Softmax(dim=2)
                else:
                    self.avg_pool = nn.AdaptiveAvgPool2d(1)
                if 'channel_add' in fusion_types:
                    self.channel_add_conv = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        nn.LayerNorm([self.planes, 1, 1]),
                        nn.ReLU(inplace=True),  # yapf: disable
                        nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
                else:
                    self.channel_add_conv = None
                if 'channel_mul' in fusion_types:
                    self.channel_mul_conv = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                        nn.LayerNorm([self.planes, 1, 1]),
                        nn.ReLU(inplace=True),  # yapf: disable
                        nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
                else:
                    self.channel_mul_conv = None

            def spatial_pool(self, x):
                batch, channel, height, width = x.size()
                if self.pooling_type == 'att':
                    input_x = x
                    # [N, C, H * W]
                    input_x = input_x.view(batch, channel, height * width)
                    # [N, 1, C, H * W]
                    input_x = input_x.unsqueeze(1)
                    # [N, 1, H, W]
                    context_mask = self.conv_mask(x)
                    # [N, 1, H * W]
                    context_mask = context_mask.view(batch, 1, height * width)
                    # [N, 1, H * W]
                    context_mask = self.softmax(context_mask)
                    # [N, 1, H * W, 1]
                    context_mask = context_mask.unsqueeze(-1)
                    # [N, 1, C, 1]
                    context = torch.matmul(input_x, context_mask)
                    # [N, C, 1, 1]
                    context = context.view(batch, channel, 1, 1)
                else:
                    # [N, C, 1, 1]
                    context = self.avg_pool(x)

                return context

            def forward(self, x):
                # [N, C, 1, 1]
                context = self.spatial_pool(x)

                out = x
                if self.channel_mul_conv is not None:
                    # [N, C, 1, 1]
                    channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
                    out = out * channel_mul_term
                if self.channel_add_conv is not None:
                    # [N, C, 1, 1]
                    channel_add_term = self.channel_add_conv(context)
                    out = out + channel_add_term

                return out


if __name__ == '__main__':
            model = GlobalContextBlock(inplanes=16, ratio=0.25)
            print(model)

            input = torch.randn(1, 16, 64, 64)
            out = model(input)
            print(out.shape)