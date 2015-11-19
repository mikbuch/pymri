import Tkinter as tk
import tkFont
import tkFileDialog

# TODO: Create working area frame, menu under it
# TODO: Scrolling left and right in the entry
# TODO: Wider entry
# TODO: Ctrl+a to select all in entry

def check_ev_frame():
    if var_ev.get():
        ev.grid(row=0, column=1, columnspan=10, sticky='W', \
                     padx=10, pady=10, ipadx=10, ipady=10)
        evChk['text']=""
    else:
        ev.grid_forget()
        evChk['text']="Explanatory Variables (EVs)"

def askdir_input():
    dir_chosen = tkFileDialog.askdirectory()
    var_input.set(dir_chosen)

def askdir_output():
    dir_chosen = tkFileDialog.askdirectory()
    var_output.set(dir_chosen)

def perform():
    # TODO: it is better to create a list of commands
    # because program can go through all code in search for errors
    if var_ev.get():
        from pymri.ev_conditions import get_attributes
        var_tr = tr_entry.get()
        input_dir = var_input.get()
        output_dir = var_output.get()
        get_attributes(input_dir=input_dir, tr=var_tr, output_dir=output_dir)



if __name__ == '__main__':
    form = tk.Tk()
    form.geometry('{}x{}'.format(1700, 500))
    font_size=12
    standard_font = tkFont.Font(size=font_size)

    getFld = tk.IntVar()
    var_ev = tk.IntVar() 

    var_input = tk.StringVar(None)
    var_output = tk.StringVar(None)

    form.wm_title('PyMRI main window')


    evChk = tk.Checkbutton(form, \
               text="Explanatory Variables (EVs)", onvalue=1, offvalue=0,
               command=check_ev_frame, variable=var_ev, font=standard_font,
               )
    evChk.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky='W')

    ev = tk.LabelFrame(
        form, borderwidth = font_size/6,
        text=" Explanatory Variables (EVs): ", font=standard_font,
        )


    ###########################################################################
    #
    #     EV FRAME
    #
    ###########################################################################

    # Input ###################################################################
    inFileLbl = tk.Label(
        ev, text="Select input directory:", font=standard_font
        )
    inFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=2)

    inFileTxt = tk.Entry(
        ev, width=40, font=standard_font, textvariable=var_input
        )
    inFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=3)

    inFileBtn = tk.Button(
        ev,
        command=askdir_input,
        text="Browse ...", font=standard_font
        )
    inFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=2)

    # Output ###################################################################
    outFileLbl = tk.Label(
        ev, text="Select output directory:", font=standard_font
        )
    outFileLbl.grid(row=1, column=0, columnspan=2, sticky='E', padx=5, pady=2)

    outFileTxt = tk.Entry(
        ev, width=40, font=standard_font, textvariable=var_output
        )
    outFileTxt.grid(row=1, column=2, columnspan=8, sticky="WE", pady=2)

    outFileBtn = tk.Button(
        ev,
        command=askdir_output,
        text="Browse ...", font=standard_font
        )
    outFileBtn.grid(row=1, column=10, sticky='W', padx=5, pady=2)

    tr_label = tk.Label(ev, text="TR (s):", font=standard_font)
    tr_label.grid(row=2, column=0, sticky='E', padx=5, pady=2)

    tr_entry = tk.Entry(ev, width=4, font=standard_font)
    tr_entry.grid(row=2, column=1, sticky='W', pady=2)


    ###########################################################################
    #
    #     MENU
    #
    ###########################################################################

    go = tk.Button(
        form,
        command=perform,
        text=" Go ", font=standard_font
        )
    go.grid(row=1, column=0, padx=10, pady=10)

    save = tk.Button(
        form,
        command=askdir_output,
        text="Save", font=standard_font
        )
    save.grid(row=1, column=1, padx=10, pady=10)

    load = tk.Button(
        form,
        command=askdir_output,
        text="Load", font=standard_font
        )
    load.grid(row=1, column=2, padx=10, pady=10)

    help = tk.Button(
        form,
        command=askdir_output,
        text="Help", font=standard_font
        )
    help.grid(row=1, column=3, padx=10, pady=10)

    quit = tk.Button(
        form,
        command=askdir_output,
        text="Quit", font=standard_font
        )
    quit.grid(row=1, column=4, padx=10, pady=10)


    # # STEP TWO
    # stepTwo = tk.LabelFrame(form, text=" 2. Enter Table Details: ")
    # stepTwo.grid(row=2, columnspan=7, sticky='W', \
                 # padx=5, pady=5, ipadx=5, ipady=5)

    # outTblLbl = tk.Label(stepTwo, \
          # text="Enter the name of the table to be used in the statements:")
    # outTblLbl.grid(row=3, column=0, sticky='W', padx=5, pady=2)

    # outTblTxt = tk.Entry(stepTwo)
    # outTblTxt.grid(row=3, column=1, columnspan=3, pady=2, sticky='WE')

    # fldLbl = tk.Label(stepTwo, \
                           # text="Enter the field (column) names of the table:")
    # fldLbl.grid(row=4, column=0, padx=5, pady=2, sticky='W')

    # getFldChk = tk.Checkbutton(stepTwo, \
                           # text="Get fields automatically from input file",\
                           # onvalue=1, offvalue=0)
    # getFldChk.grid(row=4, column=1, columnspan=3, pady=2, sticky='WE')

    # fldRowTxt = tk.Entry(stepTwo)
    # fldRowTxt.grid(row=5, columnspan=5, padx=5, pady=2, sticky='WE')

    # transChk = tk.Checkbutton(stepThree, \
               # text="Enable Transaction", onvalue=1, offvalue=0,
               # )
    # transChk.grid(row=6, sticky='W', padx=5, pady=2)


    # # STEP THREE
    # stepThree = tk.LabelFrame(form, text=" 3. Configure: ")
    # stepThree.grid(row=3, columnspan=7, sticky='W', \
                   # padx=5, pady=5, ipadx=5, ipady=5)


    # transRwLbl = tk.Label(stepThree, \
                 # text=" => Specify number of rows per transaction:")
    # transRwLbl.grid(row=6, column=2, columnspan=2, \
                    # sticky='W', padx=5, pady=2)

    # transRwTxt = tk.Entry(stepThree)
    # transRwTxt.grid(row=6, column=4, sticky='WE')


    form.mainloop()
