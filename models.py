import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc = 3, output_nc = 3, n_residual_blocks=9):
        super(Generator, self).__init__()
        gf_dim = 64
        self.iniBlock = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(input_nc, 64, 7),
                                      nn.InstanceNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.downsampling = nn.Sequential(nn.Conv2d(gf_dim, gf_dim * 2, 3, 2, padding=1),
                                          nn.InstanceNorm2d(gf_dim * 2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, 2, padding=1),
                                          nn.InstanceNorm2d(gf_dim * 4),
                                          nn.ReLU(inplace=True))
        self.resBlock = nn.Sequential(ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4),
                                      ResidualBlock(gf_dim * 4))
        self.upsampling = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(gf_dim * 4, gf_dim * 2, 5, 1, 2),
                                        nn.InstanceNorm2d(gf_dim * 2),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2),
                                        nn.Conv2d(gf_dim * 2, gf_dim, 5, 1, 2),
                                        nn.InstanceNorm2d(gf_dim),
                                        nn.ReLU(inplace=True))
        self.out_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(gf_dim, output_nc, 7),
                                       nn.Tanh())

    def forward(self, x):
        x = self.iniBlock(x)
        x = self.downsampling(x)
        x = self.resBlock(x)
        x = self.upsampling(x)
        x = self.out_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc = 3):
        super(Discriminator, self).__init__()

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.rgb_to_feat_2 = nn.Conv2d(3, 64, 1)
        self.rgb_to_feat_4 = nn.Conv2d(3, 128, 1)
        self.rgb_to_feat_8 = nn.Conv2d(3, 256, 1)

        self.layer_1 = nn.Sequential(nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.layer_2 = nn.Sequential(nn.Conv2d(64 * 2, 128, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(128),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.layer_3 = nn.Sequential(nn.Conv2d(128 * 2, 256, 4, stride=2, padding=1),
                                     nn.InstanceNorm2d(256),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.layer_4 = nn.Sequential(nn.Conv2d(256 * 2, 512, 4, padding=1),
                                     nn.InstanceNorm2d(512),
                                     nn.LeakyReLU(0.2, inplace=True))

        # FCN classification layer for gan(True False)
        self.gan_cls = nn.Conv2d(512, 1, 4, padding=1)

        # FCN classification layer for Rotation(0° 90° 180° 270°)
        self.rot_cls = nn.Conv2d(512, 4, 4, padding=1)
        # self.rot_cls = nn.Linear(512, 4)

    def forward(self, input):
        # 下采样
        input_down_2 = self.downsample(input)
        input_down_4 = self.downsample(input_down_2)
        input_down_8 = self.downsample(input_down_4)

        # 通道对齐
        input_down_feat_1 = self.rgb_to_feat_2(input_down_2)
        input_down_feat_2 = self.rgb_to_feat_4(input_down_4)
        input_down_feat_3 = self.rgb_to_feat_8(input_down_8)

        feat_1 = self.layer_1(input)
        feat_cat_1 = torch.cat([feat_1, input_down_feat_1], dim=1)

        feat_2 = self.layer_2(feat_cat_1)
        feat_cat_2 = torch.cat([feat_2, input_down_feat_2], dim=1)

        feat_3 = self.layer_3(feat_cat_2)
        feat_cat_3 = torch.cat([feat_3, input_down_feat_3], dim=1)

        feat_4 = self.layer_4(feat_cat_3)

        gan = self.gan_cls(feat_4)
        rot = self.rot_cls(feat_4)

        # gan_logits = F.avg_pool2d(gan, gan.size()[2:]).view(gan.size()[0], -1)
        rot_logits = F.avg_pool2d(rot, rot.size()[2:]).view(rot.size()[0], -1)

        return gan, rot_logits


class OPNet(nn.Module):
    def __init__(self):
        super(OPNet, self).__init__()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.content_branch = nn.Sequential(nn.Conv2d(512, 256, 1, 1),
                                            nn.Conv2d(256, 1, 1, 1))
        self.domain_branch = nn.Sequential(nn.Conv2d(1024, 512, 1, 1),
                                           nn.Conv2d(512, 256, 1, 1),
                                           nn.Conv2d(256, 3, 1, 1))

    def forward(self, imgA, imgB, pathA, pathB):
        # pathA = get_path(imgA.detach(), pathA_idx, grid_size=grid_size)
        # pathB = get_path(imgB.detach(), pathB_idx)
        cA = self.content_encoder(pathA)
        cB = self.content_encoder(pathB)
        sA = self.style_encoder(imgA)   # 1, 512, 1, 1
        sB = self.style_encoder(imgB)   # 1, 512, 1, 1

        pA = F.interpolate(self.content_branch(cA), size=pathA.shape[2:])
        pB = F.interpolate(self.content_branch(cB), size=pathB.shape[2:])

        domain_cls = self.domain_branch(torch.cat((sA, sB), dim=1))
        domain_cls = domain_cls.view(domain_cls.size()[0], -1)
        return pA, pB, domain_cls


class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=512):
        super(StyleEncoder, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1),
                                    nn.LeakyReLU(inplace=True))
        self.output = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(512, style_dim, 1, 1, 0))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)

        return x


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

if __name__ == '__main__':
    input = torch.rand((4 * 1, 3, 256, 256))
    Dis = Discriminator(input_nc=3)
    gan_logits, rot_logits = Dis(input)
    print(rot_logits.shape)
    print(rot_logits)
    # criterion_GAN = torch.nn.MSELoss()
    #
    # # target_real = Variable(Tensor(4).fill_(1.0), requires_grad=False)
    # target_real = torch.ones((4))
    # print(target_real)
    # gan_logits = torch.zeros((4))
    # print(gan_logits)
    #
    # print(criterion_GAN(gan_logits, target_real))
    # print(rot_logits)
    # print(F.softmax(rot_logits))
    # print(F.sigmoid(rot_logits))
