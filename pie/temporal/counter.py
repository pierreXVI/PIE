r"""
The temporal methods can display a progress bar through the use of the ``verbose`` parameter.
This bar is managed in the module ``temporal.counter``.
"""

from __future__ import print_function, division

import os


class Counter:
    """
    Class used to display a progress bar. Rewrites on the same line to get a moving progress bar.
    Prints only if there is a difference with the previous print, not to slow the bigger process.
    The counter can be called, with an integer ``i``. ``i == -1`` correspond to 0% and ``i == n - 1`` to 100%.

    :param str text: The text of the counter
    :param int n: The final value

    Example:
       >>> n_max = 400
       >>> c = Counter('Text', n_max)
       >>> for i in range(-1, n_max):
       ...     c(i)  # Will display a new message only when i = -1, 3, 7, ..., 399 in this example
       >>>
    """

    def __init__(self, text, n):
        self.text = text
        self.n = n
        self.i = -1
        if self.text:
            self.string = '\r{0} - |{{0:<{{bar_length}}}}| {{1:=4.0%}}'.format(self.text)
        else:
            self.string = '\r|{0:<{bar_length}}| {1:=4.0%}'

    def __call__(self, i):
        i += 1
        if self.n == 0 or i > self.n or 100 * self.i // self.n == 100 * i // self.n:
            return
        self.i = i
        try:
            terminal_width = os.get_terminal_size()[0]
        except (OSError, AttributeError):
            terminal_width = 80
        bar_length = terminal_width + 23 - len(self.string)
        print(self.string.format(int(bar_length * i / self.n) * '-', i / self.n, bar_length=bar_length), end='')
        if i == self.n:
            print()
