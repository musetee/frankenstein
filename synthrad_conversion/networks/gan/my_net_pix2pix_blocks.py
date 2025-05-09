from torch import nn
import torch
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - new_shape[2] // 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] // 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    Outputs:
        x: with shape (batch size, input_channels * 2, height / 2, width / 2)
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x) # (batch size, input_channels * 2, height, width)
        if self.use_bn: 
            x = self.batchnorm(x)
        if self.use_dropout: # (batch size, input_channels * 2, height, width)
            x = self.dropout(x)
        x = self.activation(x) # (batch size, input_channels * 2, height, width)
        x = self.conv2(x) # (batch size, input_channels * 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x) # (batch size, input_channels * 2, height, width)
        if self.use_dropout:
            x = self.dropout(x) # (batch size, input_channels * 2, height, width)
        x = self.activation(x) # (batch size, input_channels * 2, height, width)
        x = self.maxpool(x) # (batch size, input_channels * 2, height / 2, width / 2)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x) # (batch size, input_channels // 2, height, width)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1) # (batch size, input_channels, height, width)
        x = self.conv2(x) # (batch size, input_channels // 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x) # (batch size, input_channels // 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x) # (batch size, input_channels // 2, height, width)
        return x 

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
    
class pix2pix_UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(pix2pix_UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand0 = ExpandingBlock(hidden_channels * 64)
        self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn) # xn
    
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
class pix2pix_Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(pix2pix_Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #self.contract5 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #self.final = nn.Conv2d(hidden_channels * 4, 1, kernel_size=1)

    def forward(self, x):
        #x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        #x5 = self.contract5(x4)
        xn = self.final(x4)
        #xn = self.final(x2)
        return xn

class ContractingBlock3D(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    Outputs:
        x: with shape (batch size, input_channels * 2, height / 2, width / 2)
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm3d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x) # (batch size, input_channels * 2, height, width)
        if self.use_bn: 
            x = self.batchnorm(x)
        if self.use_dropout: # (batch size, input_channels * 2, height, width)
            x = self.dropout(x)
        x = self.activation(x) # (batch size, input_channels * 2, height, width)
        x = self.conv2(x) # (batch size, input_channels * 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x) # (batch size, input_channels * 2, height, width)
        if self.use_dropout:
            x = self.dropout(x) # (batch size, input_channels * 2, height, width)
        x = self.activation(x) # (batch size, input_channels * 2, height, width)
        x = self.maxpool(x) # (batch size, input_channels * 2, height / 2, width / 2)
        return x

class ExpandingBlock3D(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock3D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm3d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x) # (batch size, input_channels // 2, height, width)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1) # (batch size, input_channels, height, width)
        x = self.conv2(x) # (batch size, input_channels // 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x) # (batch size, input_channels // 2, height, width)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x) # (batch size, input_channels // 2, height, width)
        return x 

class FeatureMapBlock3D(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock3D, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
class pix2pix_Discriminator3D(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(pix2pix_Discriminator3D, self).__init__()
        self.upfeature = FeatureMapBlock3D(input_channels, hidden_channels)
        self.contract1 = ContractingBlock3D(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock3D(hidden_channels * 2)
        self.contract3 = ContractingBlock3D(hidden_channels * 4)
        self.contract4 = ContractingBlock3D(hidden_channels * 8)
        self.final = nn.Conv3d(hidden_channels * 16, 1, kernel_size=1)
        #self.final = nn.Conv2d(hidden_channels * 4, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        #xn = self.final(x2)
        return xn


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.
    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(real, fake)
    #print('adv loss: ', gen_adv_loss, 'rec loss: ', gen_rec_loss, 'lambda: ', lambda_recon, 'gen loss: ', gen_adv_loss + lambda_recon * gen_rec_loss)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    return gen_loss

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def get_pix2pix(input_dim = 1, real_dim = 1, hidden_channels = 2, learning_rate_G = 1e-4,learning_rate_D=5e-5, device = 'cuda'):
    gen = pix2pix_UNet(input_dim, real_dim, hidden_channels).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate_G)
    disc = pix2pix_Discriminator(input_dim + real_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate_D)
    return gen, gen_opt, disc, disc_opt

def train_onestep_pix2pix(gen, gen_opt, disc, disc_opt,inputs,real,adv_criterion,recon_criterion,lambda_recon, update_D=True):
    ### Update discriminator ###
    disc_opt.zero_grad() # Zero out the gradient before backpropagation
    with torch.no_grad():
        fake = gen(inputs)
    #if step==1:
        #print('fake shape: ',fake.shape,'real shape: ',inputs.shape)
    if update_D:
        disc_fake_hat = disc(fake.detach(), inputs) # Detach generator
        disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
        disc_real_hat = disc(real, inputs)
        disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True) # Update gradients
        disc_opt.step() # Update optimizer
        #print('disc_fake_loss: ',disc_fake_loss,'disc_real_loss: ',disc_real_loss)

    #### Update generator ####
    gen_opt.zero_grad()
    gen_loss = get_gen_loss(gen, disc, real, condition=inputs, adv_criterion=adv_criterion, recon_criterion=recon_criterion, lambda_recon=lambda_recon)
    gen_loss.backward() # Update gradients
    gen_opt.step() # Update optimizer
    
    disc_loss_onestep = disc_loss.item()
    gen_loss_onestep = gen_loss.item()
    return gen_loss_onestep, disc_loss_onestep, fake