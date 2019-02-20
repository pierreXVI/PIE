### How to install:

This module can be installed either from a local or from a remote version of the git repository.

##### With `pip`:

- `pip install git+https://github.com/pierreXVI/PIE#egg=PIE` to install from remote (will clone a local version)

- `pip install /path/to/local/repository/` to install from local

Add the `-e` option to install in editable mode.

##### Without `pip`:

- `python install /path/to/local/repository/setup.py install`

### Build the doc:

To build this project's documentation, *sphinx* need to be installed.
If the *read-the-doc* theme is installed, it will be used as the html theme.

To then build the doc:
~~~
cd /path/to/local/repository/doc/
make # to see a list of possible output format
make html # to build the doc html format, in ./build/html
~~~
