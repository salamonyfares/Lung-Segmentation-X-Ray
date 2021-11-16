import os
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib as plt
import UNET
from io import BytesIO, StringIO

def main():
    
    UNet_model = UNET(256,256,1)
    UNet_model.load_weights('Model_Weights.h5')
    st.info(__doc__)
    file = st.file_uploader("Upload file", type = ["csv", "png","jpg"])
    show_file = st.empty()
    
    if not file:
        show_file.info("please upload a file : {}".format(' '.join(["csv", "png","jpg"])))
        return
    content = file.getvalue()
    
    if isinstance(file, BytesIO):
        show_file.image(file)
        
        imgName=st.List(content)[0]
        img=cv2.imread(imgName)
        img=cv2.resize(img,(256,256))[:,:,0]
        
        norm=np.reshape(img,(1, 256, 256, 1))
        pred=UNet_model.predict(norm)
        
        plt.imshow(np.squeez(pred))
    else:
        df = pd.read_csv(file)
        st.dataframe(df.head(2))
    file.close()
    
main()