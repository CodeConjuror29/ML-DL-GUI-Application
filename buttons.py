from tkinter import *

root = Tk()

def myClick():
    myLabel = Label(root, texxt="Look I clicked a Button!!")
    myLabel.pack()

myButton = Button(root, text="Click me", command=myClick), fg=blue
myButton.pack()
root.mainloop()