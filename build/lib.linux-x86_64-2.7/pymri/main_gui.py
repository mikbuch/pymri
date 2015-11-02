import Tkinter
import tkFont
import tkFileDialog

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
    # TODO: check if EV is checked
    if var_ev.get():
        from pymri.ev_conditions import 



if __name__ == '__main__':
    form = Tkinter.Tk()
    font_size=12
    standard_font = tkFont.Font(size=font_size)

    getFld = Tkinter.IntVar()
    var_ev = Tkinter.IntVar() 
    var_input = Tkinter.StringVar(None)
    var_output = Tkinter.StringVar(None)

    form.wm_title('PyMRI main window')

    evChk = Tkinter.Checkbutton(form, \
               text="Explanatory Variables (EVs)", onvalue=1, offvalue=0,
               command=check_ev_frame, variable=var_ev, font=standard_font,
               )
    evChk.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky='W')

    ev = Tkinter.LabelFrame(
        form, borderwidth = font_size/6,
        text=" Explanatory Variables (EVs): ", font=standard_font,
        )

    ###########################################################################
    #
    #     EV FRAME
    #
    ###########################################################################

    # Input ###################################################################
    inFileLbl = Tkinter.Label(
        ev, text="Select input directory:", font=standard_font
        )
    inFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=2)

    inFileTxt = Tkinter.Entry(
        ev, width=40, font=standard_font, textvariable=var_input
        )
    inFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=3)

    inFileBtn = Tkinter.Button(
        ev,
        command=askdir_input,
        text="Browse ...", font=standard_font
        )
    inFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=2)

    # Output ###################################################################
    outFileLbl = Tkinter.Label(
        ev, text="Select output directory:", font=standard_font
        )
    outFileLbl.grid(row=1, column=0, columnspan=2, sticky='E', padx=5, pady=2)

    outFileTxt = Tkinter.Entry(
        ev, width=40, font=standard_font, textvariable=var_output
        )
    outFileTxt.grid(row=1, column=2, columnspan=8, sticky="WE", pady=2)

    outFileBtn = Tkinter.Button(
        ev,
        command=askdir_output,
        text="Browse ...", font=standard_font
        )
    outFileBtn.grid(row=1, column=10, sticky='W', padx=5, pady=2)

    tr_label = Tkinter.Label(ev, text="TR (s):", font=standard_font)
    tr_label.grid(row=2, column=0, sticky='E', padx=5, pady=2)

    tr_entry = Tkinter.Entry(ev, width=4, font=standard_font)
    tr_entry.grid(row=2, column=1, sticky='W', pady=2)


    ###########################################################################
    #
    #     MENU
    #
    ###########################################################################

    go = Tkinter.Button(
        form,
        command=perform,
        text=" Go ", font=standard_font
        )
    go.grid(row=1, column=0, padx=10, pady=10)

    save = Tkinter.Button(
        form,
        command=askdir_output,
        text="Save", font=standard_font
        )
    save.grid(row=1, column=1, padx=10, pady=10)

    load = Tkinter.Button(
        form,
        command=askdir_output,
        text="Load", font=standard_font
        )
    load.grid(row=1, column=2, padx=10, pady=10)

    help = Tkinter.Button(
        form,
        command=askdir_output,
        text="Help", font=standard_font
        )
    help.grid(row=1, column=3, padx=10, pady=10)

    quit = Tkinter.Button(
        form,
        command=askdir_output,
        text="Quit", font=standard_font
        )
    quit.grid(row=1, column=4, padx=10, pady=10)


    # # STEP TWO
    # stepTwo = Tkinter.LabelFrame(form, text=" 2. Enter Table Details: ")
    # stepTwo.grid(row=2, columnspan=7, sticky='W', \
                 # padx=5, pady=5, ipadx=5, ipady=5)

    # outTblLbl = Tkinter.Label(stepTwo, \
          # text="Enter the name of the table to be used in the statements:")
    # outTblLbl.grid(row=3, column=0, sticky='W', padx=5, pady=2)

    # outTblTxt = Tkinter.Entry(stepTwo)
    # outTblTxt.grid(row=3, column=1, columnspan=3, pady=2, sticky='WE')

    # fldLbl = Tkinter.Label(stepTwo, \
                           # text="Enter the field (column) names of the table:")
    # fldLbl.grid(row=4, column=0, padx=5, pady=2, sticky='W')

    # getFldChk = Tkinter.Checkbutton(stepTwo, \
                           # text="Get fields automatically from input file",\
                           # onvalue=1, offvalue=0)
    # getFldChk.grid(row=4, column=1, columnspan=3, pady=2, sticky='WE')

    # fldRowTxt = Tkinter.Entry(stepTwo)
    # fldRowTxt.grid(row=5, columnspan=5, padx=5, pady=2, sticky='WE')

    # transChk = Tkinter.Checkbutton(stepThree, \
               # text="Enable Transaction", onvalue=1, offvalue=0,
               # )
    # transChk.grid(row=6, sticky='W', padx=5, pady=2)


    # # STEP THREE
    # stepThree = Tkinter.LabelFrame(form, text=" 3. Configure: ")
    # stepThree.grid(row=3, columnspan=7, sticky='W', \
                   # padx=5, pady=5, ipadx=5, ipady=5)


    # transRwLbl = Tkinter.Label(stepThree, \
                 # text=" => Specify number of rows per transaction:")
    # transRwLbl.grid(row=6, column=2, columnspan=2, \
                    # sticky='W', padx=5, pady=2)

    # transRwTxt = Tkinter.Entry(stepThree)
    # transRwTxt.grid(row=6, column=4, sticky='WE')


    form.mainloop()
