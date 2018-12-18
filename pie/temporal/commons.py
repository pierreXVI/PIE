from __future__ import print_function, division
import os


class Counter:
    """
    Class used to display a progress bar. Rewrites on the same line to get a moving progress bar.
    Prints only if there is a difference with the previous print, not to slow the bigger process.

    :param str name: The name of the counter
    :param int n: The final value

    How to use:
    >>> n_max = 400
    >>> c = Counter('Counter name', n_max)
    >>> for i in range(-1, n_max):
    ...     c(i)  # Will display a new message if i = -1, 3, 7, ..., 399 in this example
    >>>
    """

    def __init__(self, name, n):
        self.name = name
        self.n = n
        self.i = -1

    def __call__(self, i):
        i += 1
        if self.n == 0 or i > self.n or 100 * self.i // self.n == 100 * i // self.n:
            return
        self.i = i
        try:
            terminal_width = os.get_terminal_size()[0]
        except OSError:
            terminal_width = 80
        except AttributeError:
            terminal_width = 80
        bar_width = terminal_width + -len(self.name) - 10
        print('\r{0} - |{1:<{bar_width}}| {2:=4.0%}'
              .format(self.name, int(bar_width * i / self.n) * '-', i / self.n, bar_width=bar_width), end='')
        if i == self.n:
            print()
