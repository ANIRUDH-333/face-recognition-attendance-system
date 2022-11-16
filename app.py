import streamlit as st
import pandas as pd
import numpy as np
import Capture_Image
import check_camera
import Train_Image
import Recognize
import os


st.title(' Face Recognition Attendance System')
# code for streamlit buttons

if st.button("Check Camera"):
    check_camera.camer()

if st.button("Capture Faces"):
    Capture_Image.takeImages()

if st.button("Train Images"):
    Train_Image.TrainImages()

if st.button("Recognize & Attendance"):
    Recognize.recognize_attendence()

if st.button("Auto Mail"):
    os.system("py automail.py")

if st.button("Quit"):
    st.write("Thank You")
    


