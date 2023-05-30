import sys
sys.path.append('optimizers')
import string
from functools import reduce
import importlib

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLineEdit,
    QLabel,
    QFormLayout,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QToolBar,
    QSlider
)


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from optimizers.swarm_method import SwarmMethod


class MplCanvas(FigureCanvas):
    
    def __init__(self, parent=None, fig=None, width=5, height=4, dpi=100):
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):

    z_data = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setFixedSize(1050, 800)

        self.hlayout = QHBoxLayout()

        self.vlayout0 = QVBoxLayout()
        self.vlayout1 = QVBoxLayout()

        self.hlayout.addLayout(self.vlayout0)
        self.hlayout.addLayout(self.vlayout1)

        # methods init
        self.optimizer = SwarmMethod(n=30, iterations=1000, tol=0.01)

        # vlayout0 init
        width, height= 12, 12
        dpi = 100
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = MplCanvas(self, self.fig, width=width, height=height, dpi=dpi)
        self.vlayout0.addWidget(self.canvas)

        self.ledit_func = QLineEdit()
        qf_layout0 = QFormLayout()
        qf_layout0.addRow("Function", self.ledit_func)
        
        
        self.sld_itera = QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.sld_itera.setRange(0, self.optimizer.itera)   
        self.sld_itera.setPageStep(1)
        self.sld_itera.valueChanged.connect(self.slider_update)
        qf_layout0.addRow("Iteration", self.sld_itera)
        self.vlayout0.addLayout(qf_layout0) 

        # vlayout1 init
        self.vlayout1.addStretch()
        
        self.gbox_iter_buttons = QGroupBox("")
        self.vlayout1.addWidget(self.gbox_iter_buttons)        
        self.grid_iter_buttons = QGridLayout()
        self.gbox_iter_buttons.setLayout(self.grid_iter_buttons)
        
        self.start_stop_button = QPushButton("START/STOP")
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        self.grid_iter_buttons.addWidget(self.start_stop_button, 0, 0, 1, 2)
        
        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.prev_step)
        self.grid_iter_buttons.addWidget(self.prev_button, 1, 0)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_step)
        self.grid_iter_buttons.addWidget(self.next_button, 1, 1)

        self.draw_button = QPushButton("Draw")
        self.draw_button.clicked.connect(self.draw_n_init_function)
        self.grid_iter_buttons.addWidget(self.draw_button, 2, 0, 1, 2)

        self.ledit_nargs = QLineEdit()
        qf_layout1 = QFormLayout()
        qf_layout1.addRow("n args", self.ledit_nargs)
        self.ledit_nargs.setText("2")
        
        self.ledit_ndots = QLineEdit()
        qf_layout1.addRow("n dots", self.ledit_ndots)
        self.ledit_ndots.setText(str(self.optimizer.n))
        
        self.ledit_iter = QLineEdit()
        qf_layout1.addRow("n iterations", self.ledit_iter)
        self.ledit_iter.setText(str(self.optimizer.iterations))
        
        self.ledit_iter_thr = QLineEdit()
        qf_layout1.addRow("itera_thresh", self.ledit_iter_thr)
        self.ledit_iter_thr.setText(str(self.optimizer.itera_thresh))

        self.ledit_x = QLineEdit()
        qf_layout1.addRow("x init", self.ledit_x)
        self.ledit_x.setText('0.0, 0.0')
        
        self.ledit_a = QLineEdit()
        qf_layout1.addRow("a init", self.ledit_a)
        self.ledit_a.setText(str(self.optimizer.a))
        
        self.ledit_b = QLineEdit()
        qf_layout1.addRow("b init", self.ledit_b)
        self.ledit_b.setText(str(self.optimizer.b))
        
        self.ledit_shift_x = QLineEdit()
        qf_layout1.addRow("shift x", self.ledit_shift_x)
        self.ledit_shift_x.setText(str(self.optimizer.shift_x))
        
        self.ledit_shift_y = QLineEdit()
        qf_layout1.addRow("shift y", self.ledit_shift_y)
        self.ledit_shift_y.setText(str(self.optimizer.shift_y))
        
        self.ledit_tol = QLineEdit()
        qf_layout1.addRow("tolerance", self.ledit_tol)
        self.ledit_tol.setText(str(self.optimizer.tol))
        self.vlayout1.addLayout(qf_layout1)
        
        self.gbox_checkboxes = QGroupBox("")
        self.vlayout1.addWidget(self.gbox_checkboxes)        
        self.vbox_checkboxes = QVBoxLayout()
        self.gbox_checkboxes.setLayout(self.vbox_checkboxes)
        
        self.cbox_inertia = QCheckBox("Inertia")
        self.cbox_inertia.setChecked(False)
        self.cbox_inertia.stateChanged.connect(self.click_inertia)
        self.vbox_checkboxes.addWidget(self.cbox_inertia)

        self.cbox_reflect = QCheckBox("Reflection")
        self.cbox_reflect.setChecked(False)
        self.cbox_reflect.stateChanged.connect(self.click_reflect)
        self.vbox_checkboxes.addWidget(self.cbox_reflect)

        self.cbox_leader = QCheckBox("Leader")
        self.cbox_leader.setChecked(False)
        self.cbox_leader.stateChanged.connect(self.click_leader)
        self.vbox_checkboxes.addWidget(self.cbox_leader)
        
        self.cbox_fading = QCheckBox("Fading")
        self.cbox_fading.setChecked(False)
        self.cbox_fading.stateChanged.connect(self.click_fading)
        self.vbox_checkboxes.addWidget(self.cbox_fading)

        self.combox_optim = QComboBox(self)
        self.combox_optim.addItems(["classic", "annealing", "extinction", "evolution", "genetic", "cat"])
        self.vbox_checkboxes.addWidget(self.combox_optim)
        
        # toolbar init
        self.toolbar = QToolBar("Main toolbar")
        self.addToolBar(self.toolbar)
        
        self.button_act = QPushButton("Change plot", self)
        self.button_act.setStatusTip("Change plot button")
        self.button_act.clicked.connect(self.change_plot)
        self.toolbar.addWidget(self.button_act)
        
        self.vlayout1.addStretch()

        self.graph_name = "main"

        # self.setCentralWidget(self.canvas)
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.hlayout)
        self.setCentralWidget(centralWidget)

        # end window init
        self.update_plot()
        self.show()
        
        # Flag for timer
        self.is_func_init = False
        
        self.prev_next_itera = self.optimizer.itera
        
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start()
        self.timer_started = False 
    
    def click_inertia(self):
        if self.cbox_inertia.isChecked():
            self.optimizer.options.append("inertia")
        else:
            self.optimizer.options.remove("inertia")

    def click_reflect(self):
        if self.cbox_reflect.isChecked():
            self.optimizer.options.append("reflection")
        else:
            self.optimizer.options.remove("reflection")

    def click_leader(self):
        if self.cbox_leader.isChecked():
            self.optimizer.options.append("leader")
        else:
            self.optimizer.options.remove("leader")

    def click_fading(self):
        if self.cbox_fading.isChecked():
            self.optimizer.options.append("fading")
        else:
            self.optimizer.options.remove("fading")

    def draw_n_init_function(self):
        f_str = str(self.ledit_func.text())
        if f_str != "":
            n_args = int(self.ledit_nargs.text())
            n_dots = int(self.ledit_ndots.text())
            n_iterations = int(self.ledit_iter.text())
            itera_thresh = int(self.ledit_iter_thr.text())
            init_a = float(self.ledit_a.text())
            init_b = float(self.ledit_b.text())
            shift_x = float(self.ledit_shift_x.text())
            shift_y = float(self.ledit_shift_y.text())
            tolerance = float(self.ledit_tol.text())
            method = str(self.combox_optim.currentText())

            self.optimizer.n = n_dots
            self.optimizer.n_args = n_args
            self.optimizer.iterations = n_iterations
            self.optimizer.itera_thresh = itera_thresh
            self.optimizer.a = init_a
            self.optimizer.b = init_b
            self.optimizer.shift_x = shift_x
            self.optimizer.shift_y = shift_y    
            self.optimizer.tol = tolerance  
            self.optimizer.optim = method
            self.optimizer.itera = 0
            self.prev_next_itera = 0
            
            self.func = create_function(f_str, n_args)

            x_init = list(map(float, self.ledit_x.text().split(", ")))

            self.method_gen = self.optimizer.minimize(self.func, x_init)
            self.xs, self.x_best, self.inform = next(self.method_gen)

            self.x_min = self.optimizer.xmin
            self.x_max = self.optimizer.xmax
            self.y_min = self.optimizer.ymin
            self.y_max = self.optimizer.ymax

            x_data = np.linspace(self.x_min, self.x_max, 1000)
            y_data = np.flip(np.linspace(self.y_min, self.y_max, 1000))

            self.x_data = np.repeat(x_data[None, :], 1000, axis=0)
            self.y_data = np.repeat(y_data[:, None], 1000, axis=1)

            tmp_data = self.func(self.x_data, self.y_data)

            self.z_data = tmp_data

            self.update_plot()
            
            self.is_func_init = True
            # self.toggle_start_stop()

    def update_plot(self):
        # Drop off the first y element, append a new one.
        # self.ydata = self.ydata[1:] + [random.randint(0, 10)]
        # self.canvas.axes.cla()  # Clear the canvas.
        # self.canvas.axes.plot(self.xdata, self.ydata, 'r')
        # # Trigger the canvas to update and redraw.
        # self.canvas.draw()

        self.canvas.axes.cla()  # Clear the canvas.

        if self.graph_name == "main":
            if self.z_data is None:
                self.x_min = -25
                self.x_max = 15
                self.y_min = -20
                self.y_max = 20
                self.inform = "INIT"

                x_data = np.linspace(self.x_min, self.x_max, 1000)
                y_data = np.linspace(self.y_min, self.y_max, 1000)

                self.x_data = np.repeat(x_data[None, :], 1000, axis=0)
                self.y_data = np.repeat(y_data[:, None], 1000, axis=1)
                self.z_data = np.ones((1000, 1000))

                # self.canvas.axes.imshow(self.z_data)
                self.c = self.canvas.axes.pcolormesh(self.x_data, self.y_data, self.z_data, cmap='viridis')
                self.canvas.axes.figure.colorbar(self.c)

                # update time
                self.canvas.axes.set_title(f"Iterations: {self.optimizer.itera}")
                self.prev_next_itera = self.optimizer.itera
                self.sld_itera.setRange(0, self.optimizer.itera)
                self.sld_itera.setValue(self.optimizer.itera)
            else:
                self.canvas.axes.set_xticks(np.arange(self.x_min, self.x_max+1, 4))
                self.canvas.axes.set_yticks(np.arange(self.y_min, self.y_max+1, 4))
                self.canvas.axes.imshow(self.z_data, extent=[self.x_min, self.x_max, self.y_min, self.y_max])
                self.c.set_clim(vmin=self.z_data.min(), vmax=self.z_data.max())

                # update tick
                self.canvas.axes.set_title(f"Iterations: {self.optimizer.itera}")
                self.prev_next_itera = self.optimizer.itera
                self.sld_itera.setRange(0, self.optimizer.itera)
                self.sld_itera.setValue(self.optimizer.itera)
                self.ledit_ndots.setText(str(self.optimizer.n))

                self.canvas.axes.scatter(self.xs[:, 0], self.xs[:, 1], color='red')
                self.canvas.axes.scatter(self.x_best[0], self.x_best[1], color='black')

                if self.inform != "END":
                    print(self.x_best, self.func(*self.x_best))
                    self.xs, self.x_best, self.inform = next(self.method_gen)

                if self.inform == "END":
                    print(self.x_best, self.func(*self.x_best))
                    self.toggle_start_stop()
        else:
            self.canvas.axes.plot(np.arange(len(self.optimizer.fgbest_list)), self.optimizer.fgbest_list, color='green')
            
            self.canvas.axes.set_title("Global best path")
            self.ledit_ndots.setText(str(self.optimizer.n))
            
            if self.inform != "END":
                print(self.x_best, self.func(*self.x_best))
                self.xs, self.x_best, self.inform = next(self.method_gen)
            
            if self.inform == "END":
                print(self.x_best, self.func(*self.x_best))
                self.toggle_start_stop()
            
            
        # Trigger the canvas to update and redraw.
        self.canvas.draw()
      
    def change_plot(self):
        if self.inform != "INIT":
            if self.graph_name == "main":
                self.graph_name = "not_main"
                self.update_plot()
            else:
                self.graph_name = "main"
                self.update_plot()
          
    def toggle_start_stop(self):
        if self.timer_started:
            self.timer.stop()
            self.timer_started = False
        elif not self.timer_started and self.is_func_init:
            self.timer.start()
            self.timer_started = True
            
    def prev_step(self):
        if not self.timer_started and self.is_func_init and self.prev_next_itera > 0:
            self.prev_next_itera -= 1
            self.sld_itera.setValue(self.prev_next_itera)
            # Clear the canvas
            self.canvas.axes.cla()
            
            self.canvas.axes.set_xticks(np.arange(self.x_min, self.x_max+1, 4))
            self.canvas.axes.set_yticks(np.arange(self.y_min, self.y_max+1, 4))
            self.canvas.axes.imshow(self.z_data, extent=[self.x_min, self.x_max, self.y_min, self.y_max])
            self.c.set_clim(vmin=self.z_data.min(), vmax=self.z_data.max())

            # Update tick
            self.canvas.axes.set_title(f"Iterations: {self.prev_next_itera}")
            self.ledit_ndots.setText(str(self.optimizer.n))

            self.canvas.axes.scatter(self.optimizer.xs_list[self.prev_next_itera][:, 0], self.optimizer.xs_list[self.prev_next_itera][:, 1], color='red')
            self.canvas.axes.scatter(self.optimizer.gbest_list[self.prev_next_itera][0], self.optimizer.gbest_list[self.prev_next_itera][1], color='black')
            
            # Trigger the canvas to update and redraw.
            self.canvas.draw()
            
    def next_step(self):
        if not self.timer_started and self.is_func_init and self.prev_next_itera < self.optimizer.itera:
            self.prev_next_itera += 1
            self.sld_itera.setValue(self.prev_next_itera)
            # Clear the canvas
            self.canvas.axes.cla()
            
            self.canvas.axes.set_xticks(np.arange(self.x_min, self.x_max+1, 4))
            self.canvas.axes.set_yticks(np.arange(self.y_min, self.y_max+1, 4))
            self.canvas.axes.imshow(self.z_data, extent=[self.x_min, self.x_max, self.y_min, self.y_max])
            self.c.set_clim(vmin=self.z_data.min(), vmax=self.z_data.max())

            # Update tick
            self.canvas.axes.set_title(f"Iterations: {self.prev_next_itera}")
            self.ledit_ndots.setText(str(self.optimizer.n))

            self.canvas.axes.scatter(self.optimizer.xs_list[self.prev_next_itera][:, 0], self.optimizer.xs_list[self.prev_next_itera][:, 1], color='red')
            self.canvas.axes.scatter(self.optimizer.gbest_list[self.prev_next_itera][0], self.optimizer.gbest_list[self.prev_next_itera][1], color='black')
            
            # Trigger the canvas to update and redraw.
            self.canvas.draw()
            
    def slider_update(self):
        if not self.timer_started and self.is_func_init:
            self.prev_next_itera = self.sld_itera.value()
            # Clear the canvas
            self.canvas.axes.cla()
            
            self.canvas.axes.set_xticks(np.arange(self.x_min, self.x_max+1, 4))
            self.canvas.axes.set_yticks(np.arange(self.y_min, self.y_max+1, 4))
            self.canvas.axes.imshow(self.z_data, extent=[self.x_min, self.x_max, self.y_min, self.y_max])
            self.c.set_clim(vmin=self.z_data.min(), vmax=self.z_data.max())

            # Update tick
            self.canvas.axes.set_title(f"Iterations: {self.prev_next_itera}")
            self.ledit_ndots.setText(str(self.optimizer.n))

            self.canvas.axes.scatter(self.optimizer.xs_list[self.prev_next_itera][:, 0], self.optimizer.xs_list[self.prev_next_itera][:, 1], color='red')
            self.canvas.axes.scatter(self.optimizer.gbest_list[self.prev_next_itera][0], self.optimizer.gbest_list[self.prev_next_itera][1], color='black')
            
            # Trigger the canvas to update and redraw.
            self.canvas.draw()


def create_function(f_str, n_args):
    vocabulary = ['sin', 'cos', 'tan', 'exp', 'log', 'log10', 'log2', 'abs', 'sqrt', 'arcsin', 'arccos', 'arctan']
    alphabet = string.ascii_lowercase[:n_args]
    xs = reduce(lambda a, b: f'{a}, {b}', alphabet)

    for tmp_func in vocabulary:
        f_str = f_str.replace(tmp_func, f'np.{tmp_func}')
    
    with open("tmp_function.py", "w") as f:
        f.write("import numpy as np\n\ndef func({}): return {}".format(xs, f_str))

    import tmp_function
    importlib.reload(tmp_function)

    return tmp_function.func


app = QApplication(sys.argv)
w = MainWindow()
app.exec()
