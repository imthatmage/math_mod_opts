import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# VARS CONSTS:

# New figure and plot variables so we can manipulate them

_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}

dataSize = 1000  # For synthetic data

# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Helvetica 25'
sg.theme('DarkTeal12')

# layout = [[sg.Canvas(key='figCanvas')],
#           [sg.Button('Update', font=AppFont), sg.Button('Exit', font=AppFont)]]

# Define the window layout
left_column = [
    [sg.Text("Plot test")],
    [sg.Canvas(key="figCanvas")],
    [sg.HSeparator()],
    [sg.Text("Sympy Function", size =(15, 1)), sg.InputText(key="func_button")],
    [sg.Button("Submit", font=AppFont)],
]

right_column = [
    [sg.Text("Parameters")],
    [sg.Button("Update", font=AppFont)],
]

layout = [
    [
        sg.Column(left_column),
        sg.VSeparator(),
        sg.Column(right_column),
    ]
]
_VARS['window'] = sg.Window('Such Window',
                            layout,
                            finalize=True,
                            resizable=False,
                            location=(100, 100),
                            font=AppFont,
                            element_justification="right")

# \\  -------- PYSIMPLEGUI -------- //


# \\  -------- PYPLOT -------- //


def makeSynthData():
    # data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], (512, 521))
    data = np.random.normal(0, 1, 512)
    return data


def drawChart(fig, ax):
    _VARS['pltFig'] = fig
    data = makeSynthData()
    ax.plot(data)
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


# Recreate Synthetic data, clear existing figre and redraw plot.

def updateChart():
    _VARS['fig_agg'].get_tk_widget().forget()
    data = makeSynthData()
    # plt.cla()
    # plt.clf()
    # plt.plot(dataXY[0], dataXY[1], '.k')
    ax.clear()
    ax.plot(data)
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

# \\  -------- PYPLOT -------- //


fig, ax = plt.subplots(1, 1, figsize=(12, 10))

drawChart(fig, ax)

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    # New Button (check the layout) and event catcher for the plot update
    if event == 'Update':
        updateChart()
_VARS['window'].close()
