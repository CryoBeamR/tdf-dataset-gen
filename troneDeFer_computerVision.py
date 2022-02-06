from glob import glob 
import os
import pickle 
# import caer
# import canaro
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
# import gc
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import LearningRateScheduler

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
    
    # Convert in gray color
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    # Noise-reducing and edge-preserving filter
    # Added: this filter make it easier to detect class contour 
    # gray_bf=cv2.bilateralFilter(gray,11,75,75)
    # Added: this blur make it easier to detect card contoura
    # gray_mb=cv2.GaussianBlur(gray,(5,5),0)
    
    # Edge extraction
    # Added
    # edge_bf=cv2.Canny(gray_bf,100,120)
    # edge_mb=cv2.Canny(gray_mb,50,200)
    # tr,edge_bf=cv2.threshold(gray_bf,225,235,cv2.THRESH_BINARY_INV)
    b,g,r = cv2.split(img)
#    was 131 and 132
    tr,edge_mb=cv2.threshold(g,150,156,cv2.THRESH_BINARY_INV)

    # tr,edge_mb=cv2.threshold(b,15,20,cv2.THRESH_BINARY_INV)

    # tr,edge_mb=cv2.threshold(r,121,130,cv2.THRESH_BINARY_INV)
    
    
     
    # cv2.imshow("edge card", edge)
    # cv2.waitKey(0)

    # Find the contours in the edged image
    # Added
    # cnts_bf, _ = cv2.findContours(edge_bf.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_mb, _ = cv2.findContours(edge_mb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # We suppose that the contour with largest area corresponds to the contour delimiting the card
    # cnt_bf = sorted(cnts_bf, key = cv2.contourArea, reverse = True)[0]
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

    # rect_bf=cv2.minAreaRect(cnt_bf)
    # box_bf=cv2.boxPoints(rect_bf)
    # box_bf=np.int0(box_bf)
    # areaCnt=cv2.contourArea(cnt_bf)
    # areaBox=cv2.contourArea(box_bf)
    # valid=areaCnt/areaBox>0.95
    
    if valid:
        # We want transform the zone inside the contour into the reference rectangle of dimensions (cardW,cardH)
        # ((xr,yr),(wr_bf,hr_bf),thetar)=rect_bf
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
        # cv2.imshow("Canny Bilateral Filter",edge_bf)
        cv2.imshow("Threshold",edge_mb)
        edge_bgr_mb=cv2.cvtColor(edge_mb,cv2.COLOR_GRAY2BGR)
        # edge_bgr_bf=cv2.cvtColor(edge_bf,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_bgr_mb,[box_mb],0,(0,0,255),3)
        cv2.drawContours(edge_bgr_mb,[cnt_mb],0,(0,255,0),-1)
        cv2.imshow("Contour with biggest area of Median Blur",edge_bgr_mb)

        # cv2.drawContours(edge_bgr_bf,[box_bf],0,(0,0,255),3)
        # cv2.drawContours(edge_bgr_bf,[cnt_bf],0,(0,255,0),-1)
        # cv2.imshow("Contour with biggest area of Bilateral Filter",edge_bgr_bf)
        if valid:
            cv2.imshow("Alphachannel",alphachannel)
            cv2.imshow("Extracted card",imgwarp)

    return valid,imgwarp
def findHull(img, corner=refCornerHL, debug="no"):
    """
        Find in the zone 'corner' of image 'img' and return, the convex hull delimiting
        the value and suit symbols
        'corner' (shape (4,2)) is an array of 4 points delimiting a rectangular zone, 
        takes one of the 2 possible values : refCornerHL or refCornerLR
        debug=
    """
    
    kernel = np.ones((3,3),np.uint8)
    corner=corner.astype(np.int_)

    # We will focus on the zone of 'img' delimited by 'corner'
    # x1=int(corner[0][0])
    # y1=int(corner[0][1])
    # x2=int(corner[2][0])
    # y2=int(corner[2][1])
    # w=x2-x1
    # h=y2-y1
    # zone=img[y1:y2,x1:x2].copy()

    strange_cnt=np.zeros_like(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thld=cv2.Canny(gray,30,200)
    thld = cv2.dilate(thld,kernel,iterations=1)
    if debug!="no": cv2.imshow("thld",thld)
    key=cv2.waitKey(0)
    # Find the contours
    contours,_=cv2.findContours(thld.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_area=30 # We will reject contours with small area. TWEAK, 'zoom' dependant
    min_solidity=0.2 # Reject contours with a low solidity. TWEAK
    
    concat_contour=None # We will aggregate in 'concat_contour' the contours that we want to keep
    
    ok=True
    for c in contours:
        area=cv2.contourArea(c)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        # Determine the center of gravity (cx,cy) of the contour
        M=cv2.moments(c)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here is to keep only the contours which are closed to the center of the zone
        
        if debug != "no" :
            cv2.drawContours(img,[c],0,(255,0,0),-1)
        if concat_contour is None:
            concat_contour=c
        elif cv2.contourArea(concat_contour) < area:
            concat_contour=c
           
        if debug != "no" and solidity <= min_solidity :
            print("Solidity",solidity)
            cv2.drawContours(strange_cnt,[c],0,255,2)
            cv2.imshow("Strange contours",strange_cnt)
            
     
    if concat_contour is not None:
        # At this point, we suppose that 'concat_contour' contains only the contours corresponding the value and suit symbols   
        # We can now determine the hull
        hull=cv2.convexHull(concat_contour)
        hull_area=cv2.contourArea(hull)
        # If the area of the hull is to small or too big, there may be a problem
        min_hull_area=60    000 # TWEAK, deck and 'zoom' dependant
        max_hull_area=78000 # TWEAK, deck and 'zoom' dependant
        # print("Hull area=",hull_area)
        if hull_area < min_hull_area or hull_area > max_hull_area: 
            ok=False
            if debug!="no":
                print("Hull area=",hull_area,"too large or too small")
        # So far, the coordinates of the hull are relative to 'zone'
        # We need the coordinates relative to the image -> 'hull_in_img' 
        hull_in_img=hull

    else:
        ok=False
    
    
    if debug != "no" :
        if concat_contour is not None:
            cv2.drawContours(img,[hull],0,(0,255,0),1)
            cv2.drawContours(img,[hull_in_img],0,(0,255,0),1)
        cv2.imshow("Image",img)
        if ok and debug!="pause_always":
            key=cv2.waitKey(1)
        else:
            key=cv2.waitKey(0)
        if key==27:
            return None
    if ok == False:
        
        return None
    
    return hull_in_img

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

# img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
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



# video_dir="data/videos_test"
# extension="mkv"
# imgs_dir="data/cards"

# for videoname in os.listdir(video_dir):
#     card_name=videoname
#     video_fn=os.path.join(video_dir,card_name)
#     output_dir=os.path.join(imgs_dir,card_name[:-4])
#     if not os.path.isdir(output_dir):
#         os.makedirs(output_dir)
#     imgs=extract_cards_from_video(video_fn,output_dir)
#     print("Extracted images for %s : %d"%(card_name,len(imgs)))

# card_name = "Catelyn Stark"
# extract_cards_from_video(f"data/videos/{card_name}.mkv",output_dir=f"data/cards/{card_name}",debug= True)

card_name = "Robb Stark"
# extract_cards_from_video(f"data/videos/{card_name}.mkv",debug= True ,live=True)

img = cv2.imread(f"./data/cards/{card_name}/997319738.png",cv2.IMREAD_UNCHANGED)

data_dir="data"
# backgrounds_pck_fn=data_dir+"/backgrounds.pck"
# dtd_dir="dtd/images/"
# bg_images=[]
# for subdir in glob(dtd_dir+"/*"):
#     for f in glob(subdir+"/*.jpg"):
#         bg_images.append(mpimg.imread(f))
# print("Nb of images loaded :",len(bg_images))
# print("Saved in :",backgrounds_pck_fn)
# pickle.dump(bg_images,open(backgrounds_pck_fn,'wb'))

imgs_dir="data/cards"
cards_pck_fn=data_dir+"/cards.pck"

cards={}
for card_name in os.listdir(imgs_dir):
     
    card_dir=os.path.join(imgs_dir,card_name)
    if not os.path.isdir(card_dir):
        print(f"!!! {card_dir} does not exist !!!")
        continue
    cards[card_name]=[]
    for f in glob(card_dir+"/*.png"):
        img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
        hull=findHull(img,refCornerHL,debug="no") 
        if hull is None: 
            print(f"File {f} not used.")
            continue
        # We store the image in "rgb" format (we don't need opencv anymore)
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
        cards[card_name].append((img,hull))
    print(f"Nb images for {card_name} : {len(cards[card_name])}")



print("Saved in :",cards_pck_fn)
pickle.dump(cards,open(cards_pck_fn,'wb'))

cv2.destroyAllWindows()
def display_img(img,polygons=[],channels="bgr",size=9):
    """
        Function to display an inline image, and draw optional polygons (bounding boxes, convex hulls) on it.
        Use the param 'channels' to specify the order of the channels ("bgr" for an image coming from OpenCV world)
    """
    if not isinstance(polygons,list):
        polygons=[polygons]    
    if channels=="bgr": # bgr (cv2 image)
        nb_channels=img.shape[2]
        if nb_channels==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    fig,ax=plt.subplots(figsize=(size,size))
    ax.set_facecolor((0,0,0))
    ax.imshow(img)

class Cards():
    def __init__(self,cards_pck_fn=cards_pck_fn):
        self._cards=pickle.load(open(cards_pck_fn,'rb'))
        # self._cards is a dictionary where keys are card names (ex:'Kc') and values are lists of (img,hullHL,hullLR) 
        self._nb_cards_by_value={k:len(self._cards[k]) for k in self._cards}
        print("Nb of cards loaded per name :", self._nb_cards_by_value)
        
    def get_random(self, card_name=None, display=False):
        if card_name is None:
            card_name= random.choice(list(self._cards.keys()))
        card=self._cards[card_name][random.randint(0,self._nb_cards_by_value[card_name]-1)]
        if display:
            if display: display_img(card,"rgb")
        return card,card_name
    
# cards = Cards()


cv2.waitKey(0)
# findHull(img ,debug="yes")
