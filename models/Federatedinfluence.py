import six
import torch
from models.influence import grad_z,stest
import models.utility as utility
import numpy as np
from torchvision import datasets, transforms
from models.deliver import deliver


def Federatedinfluence(HOST,PORT, world_size, partyid, net,dataset,
                       bs=32,test_id=0,device=torch.device('cpu'),epoch=10,BUFSIZ=1024000000):


    if partyid==0:
        """ server """

        #初始化通信
        server=deliver(HOST,PORT,partyid=partyid,world_size=world_size,BUFSIZ=BUFSIZ)


        #加载测试数据
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)
        test_set= torch.utils.data.DataLoader(dataset_test, batch_size=bs)


        """calculate influence function"""
#        model=net.to(device)
        model=net
        test_id=test_id      #选择的test数据id
        data, target = test_set.dataset[test_id]
        data= test_set.collate_fn([data])
        target = test_set.collate_fn([target])

        print("begin grad")
        grad_test=grad_z(data,target,model,device=device,create_graph=False)   #v初始值
        print("end grad")
        v=grad_test
        s_test = []


        """server与client交互计算s_test"""
        for i in range(world_size):
            #id_client=random.randint(1,args.world_size) #选择client
            #向选择的client发送当前v

            print("send v to client:",i+1)

            #tensor to list
            v_temp = []
            for key in v:
                v_temp.append(key.cpu().numpy().tolist())

            # 拼接传输的数据内容
            data = {}
            data["data"] = v_temp
            data["partyid"] = partyid

            server.send(data,i+1)


            #当client计算完成，从client接收v，准备发给下一个client
            print("rec v from client:",i+1)
            v_new=server.rec(i+1)

            s_test.append(v_new)
	    #s_test计算结束，将最终s_test发送给全体client


        e_s_test=[]
        for i in range(len((s_test[0])['data'])):
            e_s_test.append((np.mean([np.array(((s_test[j])['data'])[i]) for j in range(world_size)], axis=0)))
            e_s_test[i]=e_s_test[i].tolist()

        server.broadcast(sendData=e_s_test)
        """交互结束"""

        print("rec influence")
        allinfluence=[]
        for i in range(world_size):
            temp=server.rec(i+1)
            allinfluence.append(temp)
        torch.save(allinfluence,'influence')
        return allinfluence

    else:
        """ client """

        #初始化通信
        client=deliver(HOST,PORT,partyid=partyid,world_size=world_size)

        #加载训练数据
        data=dataset
        bsz=bs
        train_set=torch.utils.data.DataLoader(data, batch_size=bsz)
        model=net.to(device)
        data, target= train_set.dataset[0]
        data = train_set.collate_fn([data])
        target= train_set.collate_fn([target])
        grad_v = grad_z(data, target, model,device=device)
        v=grad_v


        """calculate influence function"""

        """ 和server交互计算s_test，可以循环迭代(当前只进行了一次迭代，没有循环）"""
        v_new=(client.rec())

        # #list to tensor
        temp=v_new['data']
        print(type(temp))
        num=len(temp)
        v_=list(v)
        for i in range(num):
            v_[i]=torch.tensor(temp[i]).to(device)
        s_test=stest(v_,model,train_set,device=device,damp=0.01,scale=1000.0,repeat=5)   #计算s_test


        #向server发送s_test,进行下一次迭代

        # tensor to list
        v_temp=[]
        for temp in s_test:
            v_temp.append(temp.cpu().detach().numpy().tolist())

        # 拼接传输的数据内容
        data = {}
        data["data"] = v_temp
        data["partyid"] = partyid

        client.send(data)
        #迭代完成后，从server接收最终的s_test，计算influence function
        recData=client.rec()
        s_test_fin=recData
        num=len(s_test_fin)
        print(type(s_test_fin))
        print(num)
        for i in range(num):
            print(i)
            s_test_fin[i]=torch.tensor(s_test_fin[i]).to(device)
        """s_test计算结束，得到最终的s_test_fin，开始计算influence"""

        print("client:",partyid,"calculate influence")
        n=len(train_set.dataset)
        influence=np.array([i for i in range(n)],dtype='float32')
        for i in utility.create_progressbar(len(train_set.dataset), desc='influence', start=0):

            #计算grad
            data, target= train_set.dataset[i]
            data = train_set.collate_fn([data])
            target= train_set.collate_fn([target])
            grad_z_vec = grad_z(data, target, model,device=device)
            #计算influence
            inf_tmp = -sum([torch.sum(k * j).data.cpu().numpy() for k, j in six.moves.zip(grad_z_vec, s_test_fin)]) /n
            influence[i]=inf_tmp
#        influence=torch.tensor(influence).to(device)
        # 拼接传输的数据内容
        data = {}
        data["data"] = list(influence)
        data["partyid"] = partyid
        print(type(data))
        #向服务器发送influence
        print("client:",partyid,"send influence to server")
        client.send(data)
        print("client:",partyid,"over")