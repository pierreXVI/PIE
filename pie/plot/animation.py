import matplotlib.pyplot as plt


class Animation:
    def __init__(self, t, x, list_y, list_label=None, list_ls=None, ticks=None, repeat=True):
        self.speed = 1

        self.x = x
        self.t = t
        self.list_y = list_y
        self.list_line = len(list_y) * [None]
        if list_label is None:
            list_label = len(list_y) * [None]
        if list_ls is None:
            list_ls = len(list_y) * ['+-']
        self.repeat = repeat

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        if ticks is not None:
            self.ax.set_xticks(ticks)

        self.i = 0
        self.title = 'n_iter = {{0:0>{width1}}}, t = {{1:0>{width2}.2f}}, dt = {{2:0.2E}}' \
            .format(width1=len(str(len(self.t))), width2=len('{0:0.2f}'.format(self.t[-1])))
        for i in range(len(self.list_y)):
            self.list_line[i], = self.ax.plot(self.x, list_y[i][0], list_ls[i], label=list_label[i])

        self.ax.legend()
        self.update()
        plt.show()

    def update(self):
        try:
            dt = self.t[self.i + 1] - self.t[self.i]
        except IndexError:
            dt = 0
        self.ax.set_title(self.title.format(self.i, self.t[self.i], dt))
        for i in range(len(self.list_y)):
            self.list_line[i].set_ydata(self.list_y[i][self.i])
        self.fig.canvas.draw()

        if self.i + 1 >= len(self.t):
            if self.repeat:
                self.i = 0
                self.fig.canvas._tkcanvas.after(int(1E3 / self.speed), self.update)
                return
            else:
                return

        t0 = self.t[self.i]
        while self.t[self.i] - t0 < int(1 / self.speed):
            self.i += 1
            if self.i >= len(self.t):
                self.i -= 1
                break
        self.fig.canvas._tkcanvas.after(int(1E3 / self.speed), self.update)
