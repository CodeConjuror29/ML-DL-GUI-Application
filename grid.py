from tkinter import *

root = Tk()

# We are creating a text widget
myLabel1 = Label(root, text="Hello World!")
myLabel2 = Label(root, text="I'm Priyanka Sen!")
myLabel3 = Label(root, text="")

# We are showing it on the screen
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=5)
myLabel3.grid(row=1, column=1) 

# Start the Tkinter event loop
root.mainloop()