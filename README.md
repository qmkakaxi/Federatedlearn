# Federatedlearn
  基于pytorch和socket实现的简易联邦学习和数据质量评估系统
  
  ## 联邦学习：
  
  ```Federatedinfluence(HOST=HOST,PORT=PORT, world_size=world_size, partyid=0, net=model,
                      dataset=data,device=device)
    HOST:联邦学习server的ip
    PORT:端口号
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
  
  

