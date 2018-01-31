import load_model
import image_sampling
import torch
from torch import exp
from torch.autograd import Variable
from torch import autograd

def triplet_loss(vec_ref, vec_pos, vec_neg):

    dist_pos = torch.norm(vec_ref - vec_pos, 2)
    dist_neg = torch.norm(vec_ref - vec_neg, 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    loss = reg_dist_pos

    return loss



if __name__ == "__main__":

    model = load_model.model

    print(model)

    learning_rate = 1e-4
    triplet_samples = 100
    optimizer = torch.optim.Adadelta(model.parameters(),  lr=learning_rate,  weight_decay=0)
    model.train(False)
    triplet_samples = 100
    batch_size = 32

    #img_ref = img_variable
    #img_pos = img_variable
    #img_neg = img_variable

    for t in range(triplet_samples):
        # extract feature of t th image triplet
        # img_***[t] is tensor of image
        #y_ref = model.forward(img_ref)
        #y_pos = model.forward(img_pos)
        #y_neg = model.forward(img_var)

        y_ref, y_pos, y_neg = model.forward(image_sampling.triplet_sampling(batch_size))

        loss = triplet_loss(y_ref, y_pos, y_neg)

        print(t, loss.data[0])

        optimizer.zero_grad()

        loss.backward()

        #for param in model.parameters():
        #   param.data -= learning_rate * param.grad.data
        optimizer.step()
