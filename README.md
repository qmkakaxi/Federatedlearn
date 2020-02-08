# Federatedlearn
  基于pytorch和socket实现的简易联邦学习和数据质量评估系统(可部署在服务器和各种边缘节点）
  
  ## FederatedLearning：
  
  ### function：
  
  ```
  FederatedLearning(HOST,PORT, world_size, partyid, net,optimizer,dataset,
                      lossfunction=F.nll_loss,device=torch.device('cpu'),epoch=10,BUFSIZ=1024000000,batch_size=64,iter=5)
  ```
 ### Parameters：
  ```
    HOST:联邦学习server的ip
    PORT:可用的端口号
    world_size:client的数量
    partyid:当前的id，id为0是server
    net:神经网络模型
    optimizer:神经网络训练优化器
    epoch:总训练的迭代次数
    device:训练选择的设备
    lossfunction:损失函数
    BUFSIZ:数据传输的buffer_size
    batch_size:神经网络训练的batch_size
    iter:每个client的内循环
```
  
  ### tutorial：
  
   一个server，两个client
  #### server：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    net = CNNMnist().to(device) #定义好的网络结构
    optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    lossfunction=F.nll_loss
    partyid=0
    
    net=FederatedLearning(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=net,optimizer=optimizer,
                      dataset=data,lossfunction=lossfunction,device=device)
  ```
  #### client1：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    net = CNNMnist().to(device) #定义好的网络结构
    optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    lossfunction=F.nll_loss
    partyid=1
    
    net=FederatedLearning(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=net,optimizer=optimizer,
                      dataset=data,lossfunction=lossfunction,device=device)
  ```
   #### client2：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    net = CNNMnist().to(device) #定义好的网络结构
    optimizer=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.5)
    lossfunction=F.nll_loss
    partyid=2
    
    net=FederatedLearning(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=net,optimizer=optimizer,
                      dataset=data,lossfunction=lossfunction,device=device)
  ```
  
   ## Federatedinfluence：
  
  ### function：
  
  ```
  Federatedinfluence(HOST,PORT, world_size, partyid, net,dataset,
                       bs=32,test_id=0,device=torch.device('cpu'),BUFSIZ=1024000000)
  ```
 ### Parameters：
  ```
    HOST:联邦学习server的ip
    PORT:可用的端口号
    world_size:client的数量
    partyid:当前的id，id为0是server
    net:神经网络模型
    bs:batch_size
    device:训练选择的设备
    BUFSIZ:数据传输的buffer_size

```
  
  ### tutorial：
  
   一个server，两个client
  #### server：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    w_wag =torch.load('w_wag')
    model=CNNMnist().to(device)
    model.load_state_dict(w_wag)
    partyid=0
    influence=Federatedinfluence(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=model,
                      dataset=data,device=device)
  ```
  #### client1：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    w_wag =torch.load('w_wag')
    model=CNNMnist().to(device)
    model.load_state_dict(w_wag)
    partyid=0
    influence=Federatedinfluence(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=model,
                      dataset=data,device=device)
  ```
   #### client2：
  ```
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    HOST=127.0.0。1
    PORT=123456
    world_size=2
    w_wag =torch.load('w_wag')
    model=CNNMnist().to(device)
    model.load_state_dict(w_wag)
    partyid=0
    influence=Federatedinfluence(HOST=HOST,PORT=PORT, world_size=world_size, partyid=partyid, net=model,
                      dataset=data,device=device)
  ```

