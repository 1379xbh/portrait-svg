"""Conservative landmark-guided simplification of clean portrait artwork."""
import cv2
import numpy as np
from portrait_regions import detect_regions
from line_geometry import spline_path


def simplify(gray):
    regions=detect_regions(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB))
    if regions is None:
        return gray.copy(), {'face_detected':False,'simplified_brows':0,'texture_cells':0}
    labels,p,fw=regions['labels'],regions['points'],regions['face_width']
    ink=(gray<185).astype(np.uint8)*255
    result=gray.copy()
    stroke=max(2,round(fw*.006))
    brows=0
    _,ink_components,ink_stats,_=cv2.connectedComponentsWithStats((ink>0).astype(np.uint8),8)
    for label in (12,13):
        mask=(labels==label).astype(np.uint8)*255
        candidates=np.bincount(ink_components[mask>0]);candidates[0]=0
        for idx in np.argsort(candidates)[::-1]:
            if not candidates[idx]:break
            x,y,w,h,area=ink_stats[idx]
            if fw*.12<w<fw*.5 and h<fw*.16 and area<fw*fw*.045:
                mask=(ink_components==idx).astype(np.uint8)*255
                break
        local=cv2.bitwise_and(ink,mask)
        size=max(3,round(fw*.02))
        merged=cv2.morphologyEx(local,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
        contours,_=cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)<fw*fw*.001:continue
        # Keep original brow shape envelope; remove the hatch marks inside it.
        contour=cv2.convexHull(contour)
        points=contour[:,0,:].astype(float)
        _,smooth=spline_path(np.vstack([points,points[0]]),True,fw*.006,return_samples=True)
        erase=np.zeros_like(mask)
        cv2.drawContours(erase,[contour],-1,255,cv2.FILLED)
        erase=cv2.dilate(erase,np.ones((5,5),np.uint8))
        result[erase>0]=255
        cv2.polylines(result,[np.rint(smooth).astype(np.int32)],True,0,max(2,stroke-1),cv2.LINE_AA)
        brows+=1
    # Merge small enclosed pattern cells within the central garment area only.
    white=(ink==0).astype(np.uint8)
    n,components,stats,centers=cv2.connectedComponentsWithStats(white,8)
    texture=np.zeros_like(gray)
    count=0
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if (centers[i,1]>p[152,0]+fw*.23 and abs(centers[i,0]-p[1,1])<fw*.23
            and 30<area<fw*fw*.025 and w<fw*.35 and h<fw*.3
            and x>0 and y>0 and x+w<gray.shape[1] and y+h<gray.shape[0]):
            texture[components==i]=255;count+=1
    if count>=3:
        size=max(3,round(fw*.022))
        merged=cv2.morphologyEx(texture,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
        # Fill merged holes, but retain the exterior ink border of the tie.
        contours,_=cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)>fw*fw*.005:
                cv2.drawContours(result,[contour],-1,255,cv2.FILLED)
        # Pattern cells can continue beyond the bottom crop. Preserve the two
        # garment boundary strokes while clearing their central texture.
        ys,xs=np.where(texture>0)
        xmin=max(0,int(xs.min())-stroke*3);xmax=min(gray.shape[1],int(xs.max())+stroke*3)
        for y in range(int(ys.min()+fw*.2),gray.shape[0]):
            row=np.flatnonzero(ink[y,xmin:xmax])
            if len(row)>1 and fw*.06<row[-1]-row[0]<fw*.42:
                result[y,xmin+row[0]+stroke+1:xmin+row[-1]-stroke]=255
    # Disconnected short hair marks above the brows are optional texture.
    n,components,stats,_=cv2.connectedComponentsWithStats((result<185).astype(np.uint8),8)
    brow_y=min(p[70,0],p[300,0])
    removed=0
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if y+h<brow_y-fw*.05 and max(w,h)<fw*.16 and area<fw*fw*.001:
            erase=cv2.dilate((components==i).astype(np.uint8),np.ones((3,3),np.uint8))
            result[erase>0]=255;removed+=1
    return result,{'face_detected':True,'simplified_brows':brows,'texture_cells':count,'removed_hair_fragments':removed}
