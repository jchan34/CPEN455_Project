'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label(model, model_input, device):

    # Init tensor of probabilities to concat and compare later
    probs_batch = torch.zeros((model_input.shape[0],1),device=device)
    for label in range(NUM_CLASSES):
        class_labels  = torch.tensor(label).expand((model_input.shape[0],)).to(device)
        answer = model(model_input,class_labels,device)
        prob_batch = discretized_mix_logistic_loss(model_input,answer,sum_over_batch=False)
        probs_batch = torch.cat((probs_batch,prob_batch),dim=-1)
    answer = probs_batch[:,1:].argmax(1)
    return answer, probs_batch[:,1:]


def classifier(model, data_loader, device,mode,fid = 455):
    model.eval()
    acc_tracker = ratio_tracker()
    answers = []
    all_logits = np.zeros((1,4))

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        try:
            original_label = [my_bidict[item] for item in categories]
            original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
            answer,_= get_label(model, model_input, device)
            correct_num = torch.sum(answer == original_label)
            acc_tracker.update(correct_num.item(), model_input.shape[0])
        except:
            pass
        answer, batch_logits = get_label(model, model_input, device)
        all_logits = np.concatenate((all_logits,batch_logits.cpu().detach().numpy()),axis=0)
        answers.extend(answer.tolist())
    
    answers.append(fid)
    if mode != 'test': # If not test, we are computing accuracy of validation set
        return acc_tracker.get_ratio(), None
    # If test set, we are returning the logits for the test set and predictions (with FID at end to save to csv)
    return answers,all_logits[1:,:]
# End of your code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=4, help='Batch size for inference')
    parser.add_argument('-c', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    parser.add_argument('-f', '--file_name', type=str,
                        default='final_output', help='Name of output csv file')
    
    parser.add_argument('-a', '--model', type=str,
                        default='models/conditional_pixelcnn.pth', help='model param file')
    
    parser.add_argument('-q', '--nr_resnet', type=int, default=3,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=90,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-s', '--save_file', type=str,default='test_labels.csv', help='Name of output csv file')

    parser.add_argument('-g', '--fid', type=float,default=455, help='fid value')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    computed_mean = [0.5047, 0.4278, 0.3347]
    computed_std = [0.2477, 0.2290, 0.2322]
    image_net_mean = [0.485, 0.456, 0.406]
    image_net_std = [0.229, 0.224, 0.225]
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling,
                                        transforms.Normalize(mean=image_net_mean, std=image_net_std)
                                        ])

    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNNCond(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, input_channels=3, nr_logistic_mix=args.nr_logistic_mix, device=device, num_classes=4)
    
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load(args.model,map_location=device))
    model.eval()
    print('model parameters loaded')
    #Begin of your code

    acc,test_logits = classifier(model = model, data_loader = dataloader, device = device,mode = args.mode,fid=args.fid)
    '''save test_logits to npy file'''
    if args.mode == 'test':
        np.save('saved_test_logits.npy',test_logits)
        print('test_logits saved with shape: ' + str(test_logits.shape))
        import pandas as pd
        df = pd.read_csv('./data/template.csv')
        df['label'] = acc
        save_pth = "./data/" + args.save_file
        df.to_csv(save_pth, index=False)
        print(f"Saved CSV to {save_pth}")
    else:
        print(f"Accuracy: {acc}")

    #End of your code
        
        