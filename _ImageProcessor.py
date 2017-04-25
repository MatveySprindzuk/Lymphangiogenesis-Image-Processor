#!/usr/bin/python
#  NOT COMPLETED!!!! Code in process of development: Software tool for the image processing of the D2-40 stained carcinoma lymphangiogenesis images
# Matvey Sprindzuk, msprindzhuk@mail.ru. +375295671073, UIIP, Minsk, Belarus

# Do not forget to put all tools to the environment path, see http://askubuntu.com/questions/63250/how-to-set-environmental-variable-path
# Software has been tested on Linux Mint 18.1 edition.

import gi
from gi.repository import Gtk # GTK GUI module
import subprocess # built-in module for calling external software
from executor import execute # subprocess module wrapper, run "sudo pip install executor" if not have it yet

from scipy import*

#from scikit-image import*
from sklearn import*
from pandas import*
#import rpy2
#import opencv2
import mamba
from PIL import*
from PIL import Image
from numpy import*
import numpy as np
import pylab
from pylab import imshow, gray, show
from os import path
from glob import glob
import mahotas
import mahotas.features
import milk
from jug import TaskGenerator
from mamba import*
from PIL import Image
from PIL import ImageEnhance
#import gdal
import cv2
import matplotlib
#import matplotlib.pyplot.savefig
import pymorph
from matplotlib import pyplot as plt

class ImageProcessor:
    def __init__(self):
        window = Gtk.Window()
        window.set_size_request(200, 100)
        window.set_title("Lymphangiogenesis Image processor")
        window.connect("delete_event", Gtk.main_quit)

        vbox = Gtk.VBox(False, 0)
        window.add(vbox)
        

        buttonread = Gtk.Button(label="Read image", stock=None)
        buttonread.connect("clicked", self.readimage)
        vbox.pack_start(buttonread, True, True, 0)
        
        buttonpreprocess = Gtk.Button(label="Preprocess", stock=None)
        buttonpreprocess.connect("clicked", self.preprocess)
        vbox.pack_start(buttonpreprocess, True, True, 0)
        
     
        
        buttonsegment = Gtk.Button(label="Segment microvessels", stock=None)
        buttonsegment.connect("clicked", self.segment)
        vbox.pack_start(buttonsegment, True, True, 0)
        
        buttonarea = Gtk.Button(label="Compute area", stock=None)
        buttonarea.connect("clicked", self.area)
        vbox.pack_start(buttonarea, True, True, 0)
        
        buttonlargeobjects = Gtk.Button(label="Compute large objects", stock=None)
        buttonlargeobjects.connect("clicked", self.largeobjects)
        vbox.pack_start(buttonlargeobjects, True, True, 0)
        
   
        buttonsmallobjects = Gtk.Button(label="Compute small objects", stock=None)
        buttonsmallobjects.connect("clicked", self.smallobjects)
        vbox.pack_start(buttonsmallobjects, True, True, 0)
        
        buttonntexture = Gtk.Button(label="Compute Haralick features", stock=None)
        buttonntexture.connect("clicked", self.texture)
        vbox.pack_start(buttonntexture, True, True, 0)
      
             
        buttonentropy = Gtk.Button(label="Compute object distribution entropy", stock=None)
        buttonentropy.connect("clicked", self.entropy)
        vbox.pack_start(buttonentropy, True, True, 0)
        
        
        
        buttonmove = Gtk.Button(label="Move processed files")
        buttonmove.connect("clicked", self.movefiles)
        vbox.pack_start(buttonmove, True, True, 0)
        
        buttonarchive = Gtk.Button(label="Archive processed files")
        buttonarchive.connect("clicked", self.archivefiles)
        vbox.pack_start(buttonarchive, True, True, 0)
        


        button = Gtk.Button(stock=Gtk.STOCK_CLOSE)
        button.connect("clicked", Gtk.main_quit)
        vbox.pack_start(button, True, True, 0)
        window.show_all()

    def readimage (self, widget):   # main function for processing widget input and executing linux bash script
		
        photo = mahotas.imread('thyroid.png', as_grey=True)
        photo = photo.astype(np.uint8)   
        gray()
        imshow(photo)
        show()
        sys.path.append('C:/Python27/ArcGIS10.2/Scripts/') 
        import gdal_merge as gm 
        workspace="D:/Satellitendaten/rapideye/img/testregion/cannyedge/out/"
        os.chdir(workspace)

        sys.argv[1:] = ['-o', 'out.tif', 'allre1.tif']
        
        
    def preprocess  (self, widget):   # calling alignment bwa bash script
      
        #img = mahotas.imread('thyroid.png', as_grey=True)
        #img = img.astype(np.uint8)  
        img = cv2.imread('thyroid.png')
        dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
       
        plt.subplot(121),plt.imshow(img)
        plt.subplot(122),plt.imshow(dst)
        plt.savefig('Preprocessed_Denoised_Image')
        plt.show()
      
         #photo = photo.astype(np.uint8)  
        #enh = ImageEnhance.Contrast(photo)
        #h = enh.enhance(1.3).show("30% more contrast")
        # pylab.show()
        # show()
        
       # imshow(photo)
       # show()
       # mamba.statistic.getHistogram(photo)
       # mamba.measure.computeArea(photo, scale=(1.0, 1.0))
       #mamba.measure.computePerimeter(imIn, scale=(1.0, 1.0), grid=HEXAGONAL)
        
    def segment (self, widget):   # running converting and indexing bash script
      
        #img = mahotas.imread('thyroid.png', as_grey=True)
        #img = img.astype(np.uint8)

        ##segmented = mahotas.otsu(photo)
        ##f = segmented.astype(np.uint8)
        ### mahotas.imsave('segmented.png')
        
        ##mahotas.imsave('.png')
        ## savefig(segmented)
        #T_otsu = mahotas.thresholding.otsu(img)
        #seeds,_ = mahotas.label(img > T_otsu)
        #labeled = mahotas.cwatershed(img.max() - img, seeds)
        #pylab.imshow(labeled)
        #pylab.savefig('seeds.png')
        ##imshow(photo > segmented)
        #gray()
        
        dna = mahotas.imread('thyroid.png')
        dna = dna.astype(np.uint8)
        dna = dna.squeeze()
        dna = pymorph.to_gray(dna)
        #print dna.shape
        #print dna.dtype
        #print dna.max()
        #print dna.min()
        #dnaf = mahotas.gaussian_filter(dna, 8)
        #T = mahotas.thresholding.otsu(dnaf)
        #labeled, nr_objects = ndimage.label(dnaf > T)
        #print nr_objects
        #pylab.imshow(labeled)
        #pylab.jet()
        #pylab.show()
        
       
        T = mahotas.thresholding.otsu(dna)
        pylab.imshow(dna > T)
        gray()
        pylab.savefig('Segmentation_Results.png')
        
        pylab.show()
        
        
        show()
        
        
    def area (self, widget):   # 
        
        img = np.asarray(Image.open("thyroid.png").convert('L'))
        img = 1 * (img < 127)

        m,n = img.shape
        
       # use np.sum to count white pixels
        print("{} white pixels, out of {} pixels in total.".format(img.sum(), m*n))
        gray()
        pylab.savefig('Segmentation_Results_PIL.png')
        
        pylab.show(img)
        
        show()

# use slicing to count any sub part, for example from rows 300-320 and columns 400-440
       #S print("{} white pixels in rectangle.".format(img[300:320,400:440].sum()))
        
        
    def largeobjects (self, widget):   # calling variant calling with Pilon
      
        execute('bash ./call_variants_pilon.sh')
        

    def smallobjects (self, widget):   # calling SnpEff annotation engine
       execute('bash ./call_reducevcf.sh')
       
    #def entropy (self, widget):
	   #dna = mahotas.imread('thyroid.png')
       #dna = dna.squeeze()
       #dna = pymorph.to_gray(dna)
       #print dna.shape
       #print dna.dtype
       #print dna.max()
       #print dna.min()
       #dnaf = ndimage.gaussian_filter(dna, 8)
       #T = mahotas.thresholding.otsu(dnaf)
       #labeled, nr_objects = ndimage.label(dnaf > T)
       #print nr_objects
       #pylab.imshow(labeled)
       #pylab.jet()
       #pylab.show()
      
    def texture (self, widget):    
       img = mahotas.imread("thyroid.png")
       return mahotas.features.haralick(img).mean(0)
       
       
    def movefiles (self, widget):   # moving files
		
        execute('bash ./moveprocessedfilesone.sh') 
        
    def archivefiles (self, widget):   # putting processed file to Zip archive
		
        execute('bash ./archivefiles.sh')  
	    
 
 
 
    
        
    def area (self, widget):   # 
        # Read image
       im = cv2.imread("thyroid.png")
       print im
       gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
       gray = cv2.bilateralFilter(gray, 11, 17, 17)
       edged = cv2.Canny(gray, 30, 200)
       (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
       screenCnt = None
       cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 3)
       cv2.imshow("Result", edged)
      # cv2.waitKey(0)
       #detector = cv2.SimpleBlobDetector()
       # keypoints = detector.detect(im)
       #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 

       #cv2.imshow("Keypoints", im_with_keypoints)
       #cv2.waitKey(0)
        
        
    def largeobjects (self, widget):   # 
      
        execute('bash ./call_variants_pilon.sh')
        

    def smallobjects (self, widget):   # 
       execute('bash ./call_reducevcf.sh')
       
    def entropy (self, widget):
       execute('bash ./norm.sh')
           
        
    def texture (self, widget):    # 
       execute('bash ./call_snpeff.sh')
       
    def movefiles (self, widget):   # moving files
		
        execute('bash ./moveprocessedfilesone.sh') 
        
    def archivefiles (self, widget):   # putting processed file to Zip archive
		
        execute('bash ./archivefiles.sh') 
         
    def main(self):
        Gtk.main()
        
if __name__ == "__main__":
    sub = ImageProcessor()
    sub.main()
    
    

