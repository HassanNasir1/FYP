import tkinter.messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from inference import Inference
from PIL import Image, ImageTk
import copy

file = None

inference = Inference()


def browse():
    global file
    file = askopenfilename(filetypes=(("Image files (.tif)", "*.tif"),))
    process_file(file)


def percentage(output):
    return (output / 556) * 100


def evaluate_output(output):
    if 100 <= output < 200:
        return 'Early Stage - {:.4f}%'.format(percentage(output))
    if 200 <= output < 300:
        return 'Positive - {:.4f}%'.format(percentage(output))
    if 300 <= output:
        return 'Severe Case - {:.4f}%'.format(percentage(output))
    return "Negative"


def process_file(filename):
    if not filename:
        tkinter.messagebox.showerror(title="Error", message="File not loaded")
        return
    image = Image.open(filename)
    image_resize = copy.deepcopy(image).resize((300, 300))
    loaded_image = ImageTk.PhotoImage(image_resize)
    image_label.config(image=loaded_image)
    image_label.photo_ref = loaded_image
    output = inference.infer(image)
    rnn_output.set(evaluate_output(output['rnn']))
    cnn_output.set(evaluate_output(output['cnn']))


master = Tk()
master.geometry("600x600")

rnn_output = StringVar()
cnn_output = StringVar()
Label(master, text="RNN:").grid(row=2, sticky=W)
Label(master, text="CNN:").grid(row=3, sticky=W)
rnn_result = Label(master, text="", textvariable=rnn_output).grid(row=2, column=2, sticky=W)
cnn_result = Label(master, text="", textvariable=cnn_output).grid(row=3, column=2, sticky=W)
image_label = Label(master)
image_label.grid(row=4, column=3, sticky=W + N + S + E)
b = Button(master, text="Browse", command=browse)
b.grid(row=0, column=0, columnspan=2, rowspan=2, sticky=W, padx=5, pady=5)

mainloop()
