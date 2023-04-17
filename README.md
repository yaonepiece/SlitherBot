# SlitherBot

Implementation of reienforcement learning algorithm using [slither.io](http://slither.io/) game as example

## Runtime Requirement:

Python 3
- mss
- PIL
- numpy
- opencv
- tensorflow 1.12.0

```
pip install mss PIL numpy tensorflow opencv-python
```

## Custom Screen Capture API Document

- Class: ScreenCapturer
  - ScreenCapturer(x, y, cropx, cropy, outx=0, outy=0, n=4)
    - `x` and `y` specify the location of the centre anchor of the capturer in pixel
    - `cropx` and `cropy` specify the height and width of the capturer in pixel
    - `outx` and `outy` specify the prefer output height and width of the capturer in pixel
    - `n` max number of frame store in stack
  - ScreenCapturer.get_pic()
    - returns a color image of the screen
  - ScreenCapturer.get_gray() 
    - returns a gray image of the screen
  - ScreenCapture.save_pic(img)
    - returns nothing
    - `img` the image to save in the stack
  - ScreenCapture.screen_data
    - contains the image stack saved by calling save_pic
    
## Game Controller API Document

- Class: SlitherChromeController
    - SlitherChromeController(ip, port)
      - This class send and recive state about the game using websocket with custom chrome extension
      - `ip` ip of the computer where the game is running on
      - `port` the port our chrome extension is listening
    - SlitherChromeController.turn(angle)
      - `angle` the heading of the snake, range 0 to 1 where 0 is north
    - SlitherChromeController.score()
      - return the score of the game
      
```
Hello world - 2023
```
