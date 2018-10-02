import inkyphat
from PIL import ImageFont
import time

#set color and font
inkyphat.set_colour("black")
font = ImageFont.truetype(inkyphat.fonts.FredokaOne,22)

#get size of display and store it as x and y
x = (inkyphat.WIDTH)
y = (inkyphat.HEIGHT)

#show simple text in the middle of the screen
def display_txt(msg):
    inkyphat.text((x/2,y/2), msg, inkyphat.BLACK, font)
    inkyphat.show()

#displays image...
def display_img(img):
    inkyphat.set_image(Image.open(img), (x,y))

#cleans display in case the screen is burnt with images/text
def clean_display(iter):
    for i in range(iter):
        print('INFO[%i] cleaning display...' % (i))
        display_txt('')

clean_display(3)

