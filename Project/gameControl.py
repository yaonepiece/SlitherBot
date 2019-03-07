import server
import random
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer

status=0
score=0
action=0

class slitherIO(WebSocket):

    get = 0
    def handleMessage(self):
        global status, score, action
        #print(self.data)
        if self.data=='stopped':
            status=0
            score=0
        else:
            status=1
            score=int(self.data)
            self.sendMessage(f'{action}')

    def handleConnected(self):
        print('[SERVE] Got request from the web extension.')
        self.sendMessage('start')

    def handleClose(self):
        pass

def init():
    server.setUp(slitherIO)

if __name__=='__main__':
    init()
    