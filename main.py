from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
import os

# create the root window
window = Tk()
window.title('Digit Recognition AI')
window.resizable(True, True)
window.geometry('300x400')

filename = ""
directory = os.getcwd()

a = Label(window ,text = "File: ")
a.grid(row = 0, column = 0)
b = Label(window ,text = filename)
b.grid(row = 0,column = 1)

def select_file(lbl:Label):
    filename = fd.askopenfilename(
        title='Select Audio',
        initialdir=directory,
        filetypes=[('WAV files', '*.wav')]) 
    lbl.config(text=filename)

btn = ttk.Button(
    window,
    text='Select Audio',
    command= lambda : select_file(b)
).grid(row=1, column=0)

# run the application
window.mainloop()
