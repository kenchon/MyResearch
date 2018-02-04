import load_model
import image_sampling
import torch
from torch import exp
from torch.autograd import Variable
from torch import autograd
from image_sampling import triplet_sampling

def triplet_loss(feat, sim):

    dist_pos = torch.norm(feat[0] - feat[1], 2)
    dist_neg = torch.norm(feat[0] - feat[2], 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    loss = ((sim[0] - reg_dist_pos)**2 + (sim[1] - reg_dist_neg)**2)/2

    return loss


if __name__ == "__main__":
    csv_index = 0

    model = load_model.model
    learning_rate = 1e-2
    epoch = 1000
    optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate,  weight_decay=0)
    # model.train(False)    # use network as feature extractor
    batch_size = 4

    for t in range(epoch):
        try:
            csv_index = 0

            loss = 0
            row = 0
            # img, sim are list where i th column holds i th triplet
            img, sim, csv_index = triplet_sampling(batch_size, csv_index)
            #print("train.py csv_index:"+str(csv_index))

            #print("length of img\t"+str(len(img)))
            #print("length of sim\t"+str(len(sim)))

            for i in range(batch_size):
                feat = []
                for j in range(3):
                    feat.append(model.forward(Variable(img[j][i].unsqueeze_(0))))
                loss += triplet_loss(feat, (sim[0][i],sim[1][i]))


            loss /= batch_size
            print(t, loss.data[0])

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        except:
            csv_index += 0
