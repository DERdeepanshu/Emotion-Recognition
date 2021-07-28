from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
import os
from pickle import load
import run

def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir='/', title="Select an image",
                                          filetypes=(("image files", ".jpg"),
                                                     ("image files", ".png")))
    label_file_explorer.configure(text="File Opened: "+filename)

def image_recog():
    run.image_recog(filename)

def video_recog():
    run.video_recog()

window = Tk()
window.title("Emotion Recognition")
window.geometry("640x480")
window.config(background="white")

label_heading = Label(window, text="Facial Recognition", width=40, height=3, font=("Courier", 20))
label_sub_head1 = Label(window, text="Image", width=20, height=2, font=("Courier", 10))
label_sub_head2 = Label(window, text="Video", width=20, height=2, font=("courier", 10))

label_file_explorer = Label(window, text="No Image Selected",
                               width=60, height=2, fg="gray")

button_explore = Button(window, text="Browse Files", command=browseFiles, height=1, width=20)
button_predict = Button(window, text="Recognize", command=image_recog, width=10)
button_video = Button(window, text="Start", command=video_recog, width=10)
button_exit = Button(window, text="Exit", command=exit)

label_heading.place(x=0, y=0)
label_sub_head1.place(x=0, y=100)
label_file_explorer.place(x=0, y=130)
button_explore.place(x=427, y=133)
button_predict.place(x=200, y=170)
label_sub_head2.place(x=0, y=210)
button_video.place(x=200, y=250)

window.mainloop()