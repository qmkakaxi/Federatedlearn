import torch
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar
from models.Federatedinfluence import Federatedinfluence

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
    dataset='../data_of_client2'
    data = torch.load(dataset)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    #设置相关参数
    HOST=args.HOST
    PORT=args.PORT_
    world_size=3
    w_wag =torch.load('w_wag')
    model=CNNMnist().to(device)
    model.load_state_dict(w_wag)
    influence=Federatedinfluence(HOST=HOST,PORT=PORT, world_size=world_size, partyid=2, net=model,
                      dataset=data,device=device)





