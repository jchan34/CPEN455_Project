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
def sample(model, label,sample_batch_size, obs, sample_op):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        device = next(model.parameters()).device
        labels = torch.tensor(label).expand((sample_batch_size,)).to(device)
        data = data.to(device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = data
                out   = model(data_v,labels, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data
# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "mps"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int,
                        default=128, help='Batch size for inference')
    parser.add_argument('-q', '--nr_resnet', type=int, default=3,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=90,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-a', '--model', type=str,
                        default='models/conditional_pixelcnn.pth', help='model param file')
    arg = parser.parse_args()
    gen_data_dir_list = ["samples"]

    fid_score_average = 0
    ind = 0
    for gen_data_dir in gen_data_dir_list:
        if not os.path.exists(gen_data_dir):
            os.makedirs(gen_data_dir)
        paths = [gen_data_dir, ref_data_dir]
        try:
            num_imgs = min(len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir)))
            fid_score = calculate_fid_given_paths(paths, num_imgs, device, dims=192)
            print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir_list))
        except:
            fid_score = 455
            print("Dimension {:d} fails!".format(192))
            
        fid_score_average = fid_score_average + fid_score
        
    fid_score_average = fid_score_average / len(gen_data_dir_list)
    print("Average fid score: {}".format(fid_score_average))
