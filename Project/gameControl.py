import websockets
import random


class SlitherChromeController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = websockets.connect('ws://' + host + ':' + port)

    def turn(self, angle):
        self.socket.send(angle)
        return self.socket.recv()


def main():
    controller = SlitherChromeController()
    while True:
        controller.turn(random.random())


if __name__ == '__main__':
    main()
