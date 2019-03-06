import matplotlib.pyplot as plt


class Animation:
    def __init__(self, t, x, list_y, list_label=None, list_ls=None, list_lw=None, x_ticks=None, repeat=True, speed=1):
        self.t = t
        self.x = x
        self.list_y = list_y
        self.list_line = len(list_y) * [None]
        if list_label is None:
            list_label = len(list_y) * [None]
        if list_ls is None:
            list_ls = len(list_y) * ['+-']
        if list_lw is None:
            list_lw = len(list_y) * [1]
        self.repeat = repeat
        self.speed = speed
        self.run = True

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)

        self.i = 0
        self.title = 'n_iter = {{0:0>{width1}}}, t = {{1:0>{width2}.2f}}, dt = {{2:0.2E}}' \
            .format(width1=len(str(len(self.t))), width2=len('{0:0.2f}'.format(self.t[-1])))
        for i in range(len(self.list_y)):
            self.list_line[i], = self.ax.plot(self.x, list_y[i][0], list_ls[i], lw=list_lw[i], label=list_label[i])

        self.ax.legend()
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        self.update()
        plt.show()

    def update(self):
        if self.i >= len(self.t):
            if self.repeat:
                self.i = 0
            else:
                self.i = len(self.t) - 1
                self.run = False

        try:
            dt = self.t[self.i + 1] - self.t[self.i]
        except IndexError:
            dt = 0

        self.fig.suptitle('Speed x {0:0.2f}'.format(self.speed))
        self.ax.set_title(self.title.format(self.i, self.t[self.i], dt))
        for i in range(len(self.list_y)):
            self.list_line[i].set_ydata(self.list_y[i][self.i])
        self.fig.canvas.draw()

        self.i = int(self.i + max(1, self.speed))
        if self.run:
            self.fig.canvas._tkcanvas.after(int(1 / self.speed), self.update)

    def key_event(self, event):
        if event.key == '+':
            self.speed += 1
        if event.key == '-':
            self.speed = max(self.speed - 1, 1)
        if event.key == '*':
            self.speed *= 2
        if event.key == '/':
            self.speed /= 2

        if event.key == ' ':
            self.run = not self.run
            if self.run:
                self.update()
