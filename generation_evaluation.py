'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
import argparse
from torchvision import transforms
# You should modify this sample function to get the generated images from the model
# This function should save the generated images to the gen_data_dir, 
# which is fixed as 'samples/Class0', 'samples/Class1', 'samples/Class2', 'samples/Class3'
# Begin of your code
def sample_and_save(model,sample_batch_size, obs, sample_op):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        device = next(model.parameters()).device
        labels = torch.randint(low=0, high=4, size=(sample_batch_size,)).to(device)
        data = data.to(device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = data
                out   = model(data_v,labels, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    sample_t = rescaling_inv(data)
    save_images(sample_t, arg.gen_data_dir,labels=labels)
# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "mps"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-q', '--nr_resnet', type=int, default=3,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=90,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-a', '--model', type=str,
                        default='models/conditional_pixelcnn.pth', help='model param file')
    
    parser.add_argument('-g', '--generate_images', type=bool,default=False, help='Flag to generate images or not')
    parser.add_argument('-r','--gen_data_dir',type=str,default=gen_data_dir, help='Directory to save generated images')
    arg = parser.parse_args()

    for ind in range(4):
        fid_score_average = 0

        if not os.path.exists(arg.gen_data_dir):
            os.makedirs(arg.gen_data_dir)
        #Begin of your code
        # Load the model of we are generating images
        if arg.generate_images:
            model = PixelCNNCond(nr_resnet=arg.nr_resnet, nr_filters=arg.nr_filters, 
                    input_channels=3, nr_logistic_mix=arg.nr_logistic_mix, device=device, num_classes=4)
            model = model.to(device)
            sample_op = lambda x : sample_from_discretized_mix_logistic(x, arg.nr_logistic_mix)
            sample_and_save(model=model, sample_batch_size=arg.batch_size, obs =(3,32,32), sample_op=sample_op)
            
        #End of your code
        paths = [arg.gen_data_dir, ref_data_dir]
        print("#generated images: {:d}, #reference images: {:d}".format(
            len(os.listdir(arg.gen_data_dir)), len(os.listdir(ref_data_dir))))

        try:
            fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
            print("Dimension {:d} works! fid score: {}".format(192, fid_score, arg,gen_data_dir))
        except:
            fid_score = 455
            print("Dimension {:d} fails!".format(192))
            
        print("Average fid score: {}".format(fid_score))