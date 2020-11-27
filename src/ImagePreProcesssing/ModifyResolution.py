"""

@author: Nisarg Shah

Purpose: The purpose of this program is to reduce the image quality of our
         training images, to reduce the image size and speed up the computation

         We will be changing the 1080p data we received to 720p


Goals: The main goal of this program is to make sure that we can speed up the computation
       One other important aspect might be that, if we were to deploy this on a
       rasberry pie, then there are less chances that there will be 1080p camera
       this might help us in other programs and thus can be reused

"""

## Step one : Loading up our libraries

import cv2
import os
