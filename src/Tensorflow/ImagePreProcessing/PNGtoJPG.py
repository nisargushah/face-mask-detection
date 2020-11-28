"""
@author: Nisarg Shah

Purpose: To convert the PNG  files to JPEG

Goal: Changing this file extension will reduce the file size and jpg files
      compresses better which will be helpful when we are trying to move the
      data to a cloud GPU since smaller file size means faster upload time. This
      is specifically important since we will be training a lot of images and
      GPU cost are quite high.

Date: 27th November 2020

"""

###Step one importing the modules

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
        new_name = file.rsplit('.', 1)[0] + '.jpg' ## Stripping off  the .png extension and adding .jpg one
        #cv2.imshow(image)
        ## Writing a new file with the .jpg extension since we cannot rename the files
        ## Refer to footnotes to see why we cannot rename the file  using os.rename()
        try:
            cv2.imwrite(os.path.join(os.getcwd()+'/66000' ,new_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        except:
            print(file)
            cv2.imshow(image)
            ## Once we make a copy of the file with .jpg extenstion, we donot need
            ## The old file, so deleting the .png extenstion file once we are done making a .jpg copy of it
        os.remove(os.path.join(os.getcwd() + '/66000' ,file))

print("Conversion completed")   #print statement indicating that the code is completed


# Foot notes
#
#   1) We are not using PIL library is because PIL thorws and fopen error
#       To simplify things out to prevent any further conflict in the code we will be
#       using the openCV library
#
#   2) For the renaming issue, sometimes python doesn't make compatible copies of jpg
#      To minimize the issue we will be using opencv
#
#   3)  I  have hardcodedmydirectory name but you can change the name , my directory name for This
#       particular program is '/6000'. You can do Ctrl+F and replace that name
