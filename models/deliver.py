from socket import *
import numpy as np


class deliver :
    def __init__(self,HOST,PORT,partyid,world_size,BUFSIZ=1024000000):
        self.HOST=HOST
        self.PORT=PORT
        self.partyid=partyid
        self.world_size=world_size
        self.BUFSIZ=BUFSIZ
        self.PORTs  =  np.arange(self.PORT, self.PORT+ self.world_size, 1, dtype=int)

        if self.partyid==0:
        # 创建socket server，并开始监听
            self.tcpSerSocks = []
            for i in range(self.world_size):
                PORT = self.PORTs[i]
                ADDR = ('0.0.0.0', PORT)
                tcpSerSock = socket(AF_INET, SOCK_STREAM)
                tcpSerSock.bind(ADDR)
                tcpSerSock.listen(1)
                self.tcpSerSocks.append(tcpSerSock)
            print('server  waiting for connection...')

    def send(self,sendData,id=1000):

        if self.partyid==0:
        #server
            tcpSerSock, addr = self.tcpSerSocks[id-1].accept()
            tcpSerSock.sendall(str(sendData).encode())
            print('server send data to client {}'.format(id))
        else:
        #client
            PORT = self.PORTs[self.partyid-1]
            ADDR = (self.HOST, PORT)
            tcpCliSock = socket(AF_INET, SOCK_STREAM)
            tcpCliSock.connect(ADDR)
            tcpCliSock.sendall(str(sendData).encode())
            tcpCliSock.close()
            print('client {} send data to server'.format(self.partyid))



    def rec(self,id=1000):
        if self.partyid==0:
        #server
            tcpSerSock, addr = self.tcpSerSocks[id-1].accept()
            recData = ""
            while True:
                buf = tcpSerSock.recv(self.BUFSIZ)
                if not len(buf):
                    break
                else:
                    recData += buf.decode()
            recData = eval(recData)
            print('server rec data from client {}'.format(id))
        else:
        #client
            PORT = self.PORTs[self.partyid-1]
            ADDR = (self.HOST, PORT)
            tcpCliSock = socket(AF_INET, SOCK_STREAM)
            tcpCliSock.connect(ADDR)

            recData = ""
            while True:
                buf = tcpCliSock.recv(self.BUFSIZ)
                if not len(buf):
                    break
                else:
                    recData += buf.decode()
            recData = eval(recData)
            tcpCliSock.close()
            print('client {} rec data from server'.format(self.partyid))
        return recData


    def broadcast(self,sendData=''):
    #server send data to all clients
        if self.partyid==0:
        #server
            for i in range(self.world_size):
                tcpSerSock, addr = self.tcpSerSocks[i].accept()
                tcpSerSock.sendall(str(sendData).encode())
            print("server broadcast")
        else:
        #client
            PORT = self.PORTs[self.partyid-1]
            ADDR = (self.HOST, PORT)
            tcpCliSock = socket(AF_INET, SOCK_STREAM)
            tcpCliSock.connect(ADDR)

            recData = ""
            while True:
                buf = tcpCliSock.recv(self.BUFSIZ)
                if not len(buf):
                    break
                else:
                    recData += buf.decode()
            recData = eval(recData)
            tcpCliSock.close()
            return recData

    def close(self):
        if self.partyid==0:
            for i in range(self.world_size):
                self.tcpSerSocks[i].close()

