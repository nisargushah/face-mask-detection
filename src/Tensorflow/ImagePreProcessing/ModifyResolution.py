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


import os
import cv2

# Since we are in the src folder but the data is in the raw_data folder, we need to
# change the directories
dir = os.getcwd()    #Getting the current working directory

os.chdir("../")      #Moving backone directory

os.chdir(os.path.join(os.getcwd(),'raw_data/' ))  #Going into the raw_data folder

## If your python code is in the same directory as your data, comment the last two steps
#Setting the raw_data folder as out root  (root Directory)
root  = os.getcwd()

#print(root )  # Check to see if it worked. If there is any error, try uncommenting This
                  # this part out

## For loop to loop over all the directories first and then all the files
for directory, subdirectories, files in os.walk(root):
    for file in files[1:]: ## Looping through all the files in the current directory
        ## Next Step is reading the files. Refer to footnotes to see why we aren't using PIL library

        ## Sometimes imread() function throws an error if the PATH for python is not specified
        ## So os.getcwd() which gets us the current working directory is used
        ## This eliminates the need of seting up the python PATH, for this particular
        ## file structure.
        #print(os.path.join(os.getcwd()+'/66000' ,file))
        image = cv2.imread(os.path.join(os.getcwd()+'/66000' ,file))
        #print(os.getcwd())
        # Save .jpg image
        image.set(3, 1280)
        image.set(4, 720) ## Changing it to 720p



print("Conversion completed")   #print statement indicating that the code is completed
