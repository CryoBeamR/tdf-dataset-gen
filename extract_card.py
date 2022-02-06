import os 

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random


cardW=70
cardH=70
cornerXmin=0
cornerXmax=23
cornerYmin=0
cornerYmax=70

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
# You shouldn't need to change this
zoom=4
cardW*=zoom
cardH*=zoom
cornerXmin=int(cornerXmin*zoom)
cornerXmax=int(cornerXmax*zoom)
cornerYmin=int(cornerYmin*zoom)
cornerYmax=int(cornerYmax*zoom)

refCard=np.array([[0,0],[cardW,0],[cardW,cardH],[0,cardH]],dtype=np.float32)
refCardRotHL=np.array([[0,cardH],[0,0],[cardW,0],[cardW,cardH]],dtype=np.float32)
refCardRotHR=np.array([[0,0],[cardW,0],[cardW,cardH],[0,cardH]],dtype=np.float32)
refCardRotLL=np.array([[cardW,cardH],[0,cardH],[0,0],[cardW,0]],dtype=np.float32)
refCornerHL=np.array([[cornerXmin,cornerYmin],[cornerXmax,cornerYmin],[cornerXmax,cornerYmax],[cornerXmin,cornerYmax]],dtype=np.float32)
refCornerLR=np.array([[cardW-cornerXmax,cardH-cornerYmax],[cardW-cornerXmin,cardH-cornerYmax],[cardW-cornerXmin,cardH-cornerYmin],[cardW-cornerXmax,cardH-cornerYmin]],dtype=np.float32)
refCorners=np.array([refCornerHL,refCornerLR])



def give_me_filename(dirname, suffixes, prefix=""):
    """
        Function that returns a filename or a list of filenames in directory 'dirname'
        that does not exist yet. If 'suffixes' is a list, one filename per suffix in 'suffixes':
        filename = dirname + "/" + prefix + random number + "." + suffix
        Same random number for all the file name
        Ex: 
        > give_me_filename("dir","jpg", prefix="prefix")
        'dir/prefix408290659.jpg'
        > give_me_filename("dir",["jpg","xml"])
        ['dir/877739594.jpg', 'dir/877739594.xml']        
    """
    if not isinstance(suffixes, list):
        suffixes=[suffixes]
    
    suffixes=[p if p[0]=='.' else '.'+p for p in suffixes]
          
    while True:
        bname="%09d"%random.randint(0,999999999)
        fnames=[]
        for suffix in suffixes:
            fname=os.path.join(dirname,prefix+bname+suffix)
            if not os.path.isfile(fname):
                fnames.append(fname)
                
        if len(fnames) == len(suffixes): break
    
    if len(fnames)==1:
        return fnames[0]
    else:
        return fnames

def varianceOfLaplacian(img):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    Source: A.Rosebrock, https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()

def extract_card (img, output_fn=None, min_focus=120, debug=False):
    """
    """
    
    imgwarp=None
    
    # Check the image is not too blurry
    focus=varianceOfLaplacian(img)
    if focus < min_focus: 
        if debug: print("Focus too low :", focus)
        return False,None
    
    b,g,r = cv2.split(img)
#    was 131 and 132
    tr,edge_mb=cv2.threshold(g,150,156,cv2.THRESH_BINARY_INV)


    # Find the contours in the edged image
    cnts_mb, _ = cv2.findContours(edge_mb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # We suppose that the contour with largest area corresponds to the contour delimiting the card
    cnt_mb = sorted(cnts_mb, key = cv2.contourArea, reverse = True)[0]
    
    # We want to check that 'cnt' is the contour of a rectangular shape
    # First, determine 'box', the minimum area bounding rectangle of 'cnt'
    # Then compare area of 'cnt' and area of 'box'
    # Both areas sould be very close
    rect_mb=cv2.minAreaRect(cnt_mb)
    box_mb=cv2.boxPoints(rect_mb)
    box_mb=np.int0(box_mb)
    areaCnt=cv2.contourArea(cnt_mb)
    areaBox=cv2.contourArea(box_mb)
    valid=areaCnt/areaBox>0.60

    if valid:
        # We want transform the zone inside the contour into the reference rectangle of dimensions (cardW,cardH)
        ((xr,yr),(wr_mb,hr_mb),thetar)=rect_mb
        # Determine 'Mp' the transformation that transforms 'box' into the reference rectangle
        Mp=cv2.getPerspectiveTransform(np.float32(box_mb),refCardRotHL)

        # Determine the warped image by applying the transformation to the image
        imgwarp=cv2.warpPerspective(img,Mp,(cardW,cardH))
        # Add alpha layer
        imgwarp=cv2.cvtColor(imgwarp,cv2.COLOR_BGR2BGRA)
        
        # Shape of 'cnt' is (n,1,2), type=int with n = number of points
        # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
        cnta=cnt_mb.reshape(1,-1,2).astype(np.float32)
        # Apply the transformation 'Mp' to the contour
        cntwarp=cv2.perspectiveTransform(cnta,Mp)
        cntwarp=cntwarp.astype(np.int_)
        
        # We build the alpha channel so that we have transparency on the
        # external border of the card
        # First, initialize alpha channel fully transparent
        alphachannel=np.zeros(imgwarp.shape[:2],dtype=np.uint8)
        # Then fill in the contour to make opaque this zone of the card 
        cv2.drawContours(alphachannel,cntwarp,0,255,-1)
        
        # Apply the alphamask onto the alpha channel to clean it
        alphachannel=cv2.bitwise_or(alphachannel,alphamask)
        alphachannel=cv2.bitwise_and(alphachannel,alphamask)
        
        # Add the alphachannel to the warped image
        # print(imgwarp)
        imgwarp[:,:,3]=alphachannel
        
        # Save the image to file
        if output_fn is not None:
            cv2.imwrite(output_fn,imgwarp)
        
    if debug:
        cv2.imshow("Green Intensity",g)
        cv2.imshow("Red Intensity",r)
        cv2.imshow("Blue Intensity",b)
        cv2.imshow("Threshold",edge_mb)
        edge_bgr_mb=cv2.cvtColor(edge_mb,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_bgr_mb,[box_mb],0,(0,0,255),3)
        cv2.drawContours(edge_bgr_mb,[cnt_mb],0,(0,255,0),-1)
        cv2.imshow("Contour with biggest area of Median Blur",edge_bgr_mb)

        if valid:
            cv2.imshow("Alphachannel",alphachannel)
            cv2.imshow("Extracted card",imgwarp)

    return valid,imgwarp

def extract_cards_from_video(video_fn = None, output_dir=None, keep_ratio=5, min_focus=120, live=False, debug=False,):
    """
        Extract cards from media file 'video_fn' 
        If 'output_dir' is specified, the cards are saved in 'output_dir'.
        One file per card with a random file name
        Because 2 consecutives frames are probably very similar, we don't use every frame of the video, 
        but only one every 'keep_ratio' frames
        
        Returns list of extracted images
    """
    if not live :
        if not os.path.isfile(video_fn) :
            print(f"Video file {video_fn} does not exist !!!")
            return -1,[]
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    input = 1 if live else video_fn
    cap=cv2.VideoCapture(input)
    
    frame_nb=0
    imgs_list=[]
    while True:
        ret,img=cap.read()
        if not ret: break
        # Work on every 'keep_ratio' frames
        if frame_nb%keep_ratio==0:
            if output_dir is not None:
                output_fn=give_me_filename(output_dir,"png")
            else:
                output_fn=None
            valid,card_img = extract_card(img,output_fn,min_focus=min_focus,debug=debug)
            if debug: 
                k=cv2.waitKey(1)
                if k==27: break
            if valid:
                imgs_list.append(card_img)
        frame_nb+=1
    
    if debug:
        cap.release()
        cv2.destroyAllWindows()
    
    return imgs_list

bord_size=2 # bord_size alpha=0
alphamask=np.ones((cardH,cardW),dtype=np.uint8)*255
cv2.rectangle(alphamask,(0,0),(cardW-2,cardH-2),0,bord_size)
cv2.line(alphamask,(bord_size*3,0),(0,bord_size*3),0,bord_size)
cv2.line(alphamask,(cardW-bord_size*3,0),(cardW,bord_size*3),0,bord_size)
cv2.line(alphamask,(0,cardH-bord_size*3),(bord_size*3,cardH),0,bord_size)
cv2.line(alphamask,(cardW-bord_size*3,cardH),(cardW,cardH-bord_size*3),0,bord_size)
plt.figure(figsize=(10,10))
plt.imshow(alphamask)
# plt.show() 



video_dir="data/videos_test"
extension="mkv"
imgs_dir="data/cards"

for videoname in os.listdir(video_dir):
    card_name=videoname
    video_fn=os.path.join(video_dir,card_name)
    output_dir=os.path.join(imgs_dir,card_name[:-4])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    imgs=extract_cards_from_video(video_fn,output_dir)
    print("Extracted images for %s : %d"%(card_name,len(imgs)))

card_name = ""
extract_cards_from_video(f"data/videos/{card_name}.mkv",output_dir=f"data/cards/{card_name}",debug= True)


cv2.waitKey(0)
