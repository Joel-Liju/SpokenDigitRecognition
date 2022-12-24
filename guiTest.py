from tkinter import *
from tkinter import ttk
import pyaudioTest

val = None
def callback():
    global val
    val = pyaudioTest.record()
def callback2():
    pyaudioTest.sound(val)
root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
ttk.Button(frm, text="Record", command=callback).grid(column=0, row=1)
ttk.Button(frm, text="play", command=callback2).grid(column=0, row=2)
ttk.Button(frm, text="quit", command=root.destroy).grid(column=0, row=3)
root.mainloop()