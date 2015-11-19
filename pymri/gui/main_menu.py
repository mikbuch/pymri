import Tkinter as tk
import tkFont
import tkFileDialog
import os

def preprocessing():
    print('data preprocessed')

def get_ev_attributeses():
    print('attributes got')

def perform_classification():
    os.system("python classification.py")

if __name__ == '__main__':
    root = tk.Tk()
    font_size=16
    standard_font = tkFont.Font(size=font_size)

    root.wm_title('PyMRI main window')


    preproc_button = tk.Button(
        root,
        command=preprocessing,
        text="Preprocessing", font=standard_font
        )
    preproc_button.grid(
        row=0, column=0, padx=5, pady=5, ipadx=10, sticky='WE'
        )

    ev_attributes_button = tk.Button(
        root,
        command=get_ev_attributeses,
        text="EV attributeses", font=standard_font
        )
    ev_attributes_button.grid(
        row=1, column=0, padx=5, pady=5, ipadx=10, sticky='WE'
        )

    classifier_button = tk.Button(
        root,
        command=perform_classification,
        text="Classification", font=standard_font
        )
    classifier_button.grid(
        row=2, column=0, padx=5, pady=5, ipadx=10, sticky='WE'
        )

    root.mainloop()
