import numpy as np
import cv2
import matplotlib.pyplot as plt

def GenerateImage(J, T, W=6, Neighbor_TH = 0.8,D=8.25, AtomPlaneSize = [51, 51], imsize= [768,1024], FLAG_STICK_NEIGHBOR = True) :
    V = np.zeros(np.array(AtomPlaneSize)*int(D*40))

    # INITIALIZE BLOCKS
    block_keys = ['0','U','D','L','R' ,'UR', 'UL','DL','DR']
    Blocks = dict()
    for key in block_keys :
        Blocks[key] = np.zeros((int(D*40),int(D*40)))
    [X,Y]= np.meshgrid(np.linspace(-int(D*20)+1,int(D*20),int(D*20)*2),np.linspace(-int(D*20)+1,int(D*20),int(D*20)*2))
    X = X - 0.5
    Y = Y - 0.5
    Blocks['0'][np.logical_and(abs(X) <= W/2*40, abs(Y) <=W/2*40)] = 1
    Blocks['U'][np.logical_and(abs(X) <= W/2*40, Y > W/2*40)] = 1
    Blocks['D'][np.logical_and(abs(X) <= W/2*40, Y < -W/2*40)] = 1
    Blocks['L'][np.logical_and(X < -W/2*40, abs(Y) <=W/2*40)] = 1
    Blocks['R'][np.logical_and(X > W/2*40, abs(Y) <=W/2*40)] = 1
    Blocks['UR'][np.logical_and(X > W/2*40, Y <= -W/2*40)] = 1
    Blocks['UL'][np.logical_and(X < -W/2*40, Y <= -W/2*40)] = 1
    Blocks['DL'][np.logical_and(X < -W/2*40, Y > W/2*40)] = 1
    Blocks['DR'][np.logical_and(X > W/2*40, Y > W/2*40)] = 1

    for i in range(AtomPlaneSize[0]) :
        for j in range(AtomPlaneSize[1]) :
            i0 = int(D * 40 * i)
            j0 = int(D * 40 * j)
            m0 = np.copy(Blocks['0'])
            UB = False
            DB = False
            LB = False
            RB = False
            ULB = False
            URB = False
            DLB = False
            DRB = False
            
            if i <= 0 :
                UB = True
            if i >= AtomPlaneSize[0]-1 :
                DB = True
            if j == 0 :
                LB = True
            if j == AtomPlaneSize[1]-1 :
                RB = True
            '''
            if i == 0 and j == 0 :
                ULB = True
            if i == 0 and j == AtomPlaneSize[1]-1:
                URB = True
            if i == AtomPlaneSize[0]-1 and j == 0 :
                DLB = True
            if i == AtomPlaneSize[0]-1 and j == AtomPlaneSize[1]-1:
                DRB = True
            '''
            if FLAG_STICK_NEIGHBOR :
                if not UB and J[i-1,j] > Neighbor_TH : 
                    m0 += Blocks['D']
                if not DB and J[i+1,j] > Neighbor_TH :
                    m0 += Blocks['U']
                if not LB and J[i,j-1] > Neighbor_TH :
                    m0 += Blocks['L']
                if not RB and J[i,j+1] > Neighbor_TH :
                    m0 += Blocks['R']

                    
                if not UB and not LB and J[i-1,j-1] > Neighbor_TH and J[i-1,j] > Neighbor_TH and J[i,j-1] > Neighbor_TH :
                    m0 += Blocks['UL']
                if not UB and not RB and J[i-1,j+1] > Neighbor_TH and J[i-1,j] > Neighbor_TH and J[i,j+1] > Neighbor_TH :
                    m0 += Blocks['UR']
                if not DB and not LB and J[i+1,j-1] > Neighbor_TH and J[i+1,j] > Neighbor_TH and J[i,j-1] > Neighbor_TH :
                    m0 += Blocks['DL']
                if not DB and not RB and J[i+1,j+1] > Neighbor_TH and J[i+1,j] > Neighbor_TH and J[i,j+1] > Neighbor_TH :
                    m0 += Blocks['DR']
            if J[i,j] < 0 :
                m0 = 0
                for key in block_keys :
                    m0 += Blocks[key]
            V[i0:int(i0+D*40),:][:,j0:int(j0+D*40)] = m0 * J[i,j]
    # [r,c]= np.meshgrid(range(AtomPlaneSize[0]), range(AtomPlaneSize[1]))

    #T = cv2.getRotationMatrix2D((imsize[1]/2, imsize[0]/2), 8.5, 1/40)
    print("TF: ", T)
    img_transformed = cv2.warpAffine(V,T,(0,0), flags=cv2.INTER_LINEAR) 
    img_transformed = img_transformed[:imsize[0],:imsize[1]]
    print("Max : ", img_transformed.max(), "Min : ", img_transformed.min())
    #img_transformed[img_transformed<0 & img_transformed<-5]=0
    #cv2.imwrite(r'C:\Users\Junhyeok\Desktop\DashApp\GeneratePattern\test.bmp',(img_transformed*255))    
    return img_transformed

def Example() :
    imsize = [768, 1024]
    AtomPlaneSize = [51, 51]
    D = 8.25


    path = r'C:\Users\Junhyeok\Desktop\DashApp\GeneratePattern\test.bmp'

    Angle1 = -7.5
    Angle2 = 82.5
    L1 = 16.5
    L2 = 16.5
    OffsetX = 1024/2 - D*AtomPlaneSize[0]/2 * (np.cos(Angle1*np.pi/180) + np.sin(Angle1*np.pi/180) ) 
    OffsetY = 768/2 - D*AtomPlaneSize[1]/2 * (np.cos(Angle2*np.pi/180) + np.sin(Angle2*np.pi/180) )  
    A = np.array([np.cos(Angle1*np.pi/180)/40,np.sin(Angle1*np.pi/180)/40,OffsetX])
    B = np.array([np.cos(Angle2*np.pi/180)*L2/L1/40,np.sin(Angle2*np.pi/180)*L2/L1/40,OffsetY])
    T = np.vstack((A.T,B.T))
    J = np.zeros(AtomPlaneSize)
    for i in range(AtomPlaneSize[0]) :
        for j in range(AtomPlaneSize[1]) :
            midpt_i = np.round(AtomPlaneSize[0]/2) 
            midpt_j = np.round(AtomPlaneSize[1]/2)
            
            Hi = 3
            Hf = 4
            Vi = 3
            Vf = 4
            
            if(j == midpt_j) :
                J[i,j] = 1
            
            #if(j < midpt_i - Hi or j > midpt_j + Hf) :
            #    J[i,j]=0
            #if(i < midpt_i - Vi or i > midpt_i + Vf) :
            #    J[i,j]=0
    img = GenerateImage(J, T)
    cv2.imwrite(path, img*255)