import matplotlib.pyplot as plt

if plt.get_backend() != 'TkAgg':
    plt.switch_backend('TkAgg')

ANIMATION_SPEED_LOC = (0.01, 0.95)
"""The location where to plot the animation speed, in figure coordinates, floats in [0, 1]"""


class Animation:
    r"""
    Class used to plot an animation

    Press ``+`` or ``-`` to increase or decrease the animation speed.
    Press ``*`` or ``/`` to multiply or divide it by 2.
    Press ``space`` to toggle pause.
    Press ``home`` or ``end`` to set the animation at start or at end.
    Press ``left`` or ``right`` to move one step at a time.

    :param array_like t:
    :param array_like x:
    :param array_like list_y:
    :param list_label:
    :type list_label: array_like, optional
    :param list_fmt:
    :type list_fmt: array_like, optional
    :param list_lw:
    :type list_lw: array_like, optional
    :param x_ticks:
    :type x_ticks: array_like, optional
    :param title:
    :type title: str, optional
    :param repeat:
    :type repeat: bool, optional
    :param speed:
    :type speed: float, optional

    :ivar array_like t: The times steps
    :ivar array_like x: The position of the solution points
    :ivar array_like list_y: The list of curves to plot
    :ivar array_like list_line: The list of plotted matplotlib.lines.Line2D
    :ivar bool repeat: If True, restart the animation when it ends
    :ivar float speed: The animation speed
    :ivar bool run: If False, the animation is paused of stopped
    :ivar int i: The index of the current time step
    :ivar matplotlib.figure.Figure fig:
    :ivar matplotlib.axes._subplots.AxesSubplot ax:
    :ivar str _iter_text: Text to be formatted with the current iteration number, time and time step
    :ivar matplotlib.text.Text _speed_text: Text plotted to show the animation speed
    """

    def __init__(self, t, x, list_y, list_label=None, list_fmt=None, list_lw=None, x_ticks=None, title='',
                 repeat=True, speed=1):
        self.t = t
        self.x = x
        self.list_y = list_y
        self.list_line = len(list_y) * [None]
        self.repeat = repeat
        self.speed = speed
        self.run = True
        self.i = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self._iter_text = r'$N_{{{{iter}}}}$ = {{0:0>{width1}}}, $t$ = {{1:0>{width2}.2f}}, $\Delta t$ = {{2:0.2E}}' \
            .format(width1=len(str(len(self.t))), width2=len('{0:0.2f}'.format(self.t[-1])))
        self._speed_text = self.fig.text(*ANIMATION_SPEED_LOC, s='Speed x {0:0.2f}'.format(self.speed),
                                         horizontalalignment='left')

        self.ax.set_xlabel(r'$x$', fontsize='xx-large')
        self.ax.set_ylabel(r'$y$', fontsize='xx-large')
        self.ax.grid(True)
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)
        if list_label is None:
            list_label = len(list_y) * [None]
        if list_fmt is None:
            list_fmt = len(list_y) * ['+-']
        if list_lw is None:
            list_lw = len(list_y) * [1]

        for i in range(len(self.list_y)):
            self.list_line[i], = self.ax.plot(self.x, list_y[i][0], list_fmt[i], lw=list_lw[i], label=list_label[i])

        self.ax.legend(loc='upper right', ncol=2, fontsize='x-large')
        self.fig.suptitle(title, fontsize='xx-large')

        self.fig.canvas.mpl_connect('key_press_event', self._key_event)
        self._update()
        plt.show()

    def _update(self):
        if self.i < 0:
            self.i += len(self.t)
        if self.i >= len(self.t):
            if self.repeat:
                self.i = 0
            else:
                self.i = len(self.t) - 1
                self.run = False

        try:
            dt = self.t[self.i + 1] - self.t[self.i]
        except IndexError:
            dt = self.t[1] - self.t[0]

        self._speed_text.set_text('Speed x {0:0.2f}'.format(self.speed))
        self.ax.set_title(self._iter_text.format(self.i, self.t[self.i], dt), fontsize='xx-large')
        for i in range(len(self.list_y)):
            self.list_line[i].set_ydata(self.list_y[i][self.i])

            # TODO: Deal with this
            if (self.list_y[i][self.i] > 1E3).any():
                self.list_line[i].set_ydata(None)

        self.fig.canvas.draw()

        if self.run:
            self.i = int(self.i + max(1, self.speed))
            self.fig.canvas._tkcanvas.after(int(1 / self.speed), self._update)

    def _key_event(self, event):
        if event.key == '+':
            self.speed += 1
        if event.key == '-':
            self.speed = max(self.speed - 1, 1)
        if event.key == '*':
            self.speed *= 2
        if event.key == '/':
            self.speed /= 2

        if event.key == 'home':
            self.i = 0
        if event.key == 'end':
            self.i = len(self.t) - 1
        if event.key == 'left':
            self.i -= 1
        if event.key == 'right':
            self.i += 1

        if event.key == ' ':
            self.run = not self.run
            if self.run:
                self._update()

        if not self.run:
            self._update()
