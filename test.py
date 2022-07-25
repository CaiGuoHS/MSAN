import os
import time
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from models import model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

TEST_SAMPLES = './test_samples'
TEST_RESULTS = './test_results'

def save_images(images, name):
    filename = TEST_RESULTS + '/' + name
    torchvision.utils.save_image(images, filename)

def main():
    msan = model(num_resblocks=[4, 4, 4, 4], input_channels=[3, 6, 6, 6]).to(device)
    msan.load_state_dict(torch.load('./ckpts/MSAN.pth'))
    print('load deblurnet success')

    if os.path.exists(TEST_RESULTS) == False:
        os.mkdir(TEST_RESULTS)

    for images_name in os.listdir(TEST_SAMPLES):
        with torch.no_grad():
            input_image = transforms.ToTensor()(Image.open(TEST_SAMPLES + '/' + images_name).convert('RGB'))
            input_image = Variable(input_image-0.5).unsqueeze(0).to(device)

            factor = 8
            h, w = input_image.shape[2], input_image.shape[3]
            new_h, new_w = h, w
            resize = False
            if h % factor != 0:
                new_h = h + factor - h % factor
                resize = True
            if w % factor != 0:
                new_w = w + factor - w % factor
                resize = True
            if resize:
                input_image = F.pad(input_image, (0, new_w - w, 0, new_h - h), 'replicate')

            d4 = msan(input_image)[0]

            if resize:
                d4 = d4[:, :, :h, :w]
                input_image = input_image[:, :, :h, :w]

            save_images(d4+input_image+0.5, images_name)

if __name__ == '__main__':
    main()