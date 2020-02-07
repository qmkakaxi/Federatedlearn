import torch
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar
import torch.nn.functional as F
from models.FederatedLearning import FederatedLearning
from models.test import test


class Partition(object):
	""" Dataset-like object, but only access a subset of it. """
	def __init__(self, data, index):
		self.data = data
		# self.index = index
		self.index = list(index)

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]


if __name__ == '__main__':

    args = args_parser()
    dataset='../data_of_client1'
    data = torch.load(dataset)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    #设置相关参数
    HOST=args.HOST
    PORT=args.PORT_
    world_size=2
    net = CNNMnist().to(device)
    optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    lossfunction=F.nll_loss
    net=FederatedLearning(HOST=HOST,PORT=PORT, world_size=world_size, partyid=2, net=net,optimizer=optimizer,
                      dataset=data,lossfunction=lossfunction,device=device)

    test_set = torch.utils.data.DataLoader(data, batch_size=args.bs)
    args.device=device
    test_accuracy, test_loss = test(net, test_set, args)
    print("Trainng accuracy: {:.2f}".format(test_accuracy))
    print("Training loss: {:.2f}".format(test_loss))
