Development
=============================
Installing the tools
--------------------
What you'll need:
- Python3
- Git
- A text editor/IDE

On Windows:
- Python: https://www.python.org/downloads/
- Git: https://www.python.org/downloads/

On Mac:
- Python: https://www.python.org/downloads/ or alternatively using Homebrew https://brew.sh/
- Git: already installed

On Linux (Debian):
- sudo apt install git
- sudo apt install python3

The source-code for the application is hosted on the following Git repository:
https://dop-git.eng.ox.ac.uk

There are many text editors/IDEs available. A recommended free editor if you're unsure is Microsoft Visual Studio Code:
https://code.visualstudio.com/

Getting started
---------------
Once you have installed the tools above, clone the source-code from the Git repository::

    git clone https://username@dop-git.eng.ox.acu.k/diffusion/3/doptical-python.git

or::

    git clone ssh://git@dop-git.eng.ox.ac.uk:2222/diffusion/3/doptical-python.git

Create a virtual environment for development (Optional but recommended)::
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the venv module to create an isolated virtual environment. The required packages are installed using pip. Finally, the main source is installed using pip in an editable format. Changes to the source affect the application the next time it is run.
Run the following from the folder the top folder (that includes setup.py)::

	python -m venv env

Activate virtual environment before next steps:
    
Windows::

	env\Scripts\activate

Using Mac/Linux::

	source env/bin/activate

Install required modules in the virtual environment:

	pip install -r requirements.txt
	pip install -e .

To install pipython module, download PIPython-1.5.1.7 folder on Teams -> Software -> Vendor libraries, 
and change directory to within that folder. There will be a setup.py file. Run the following
from that folder::

    python setup.py install

The application can then be run in the virtual environment as follows (from the root folder):

Windows::

    python doptical\app.py

Mac/Linux::

    python doptical/app.py

To run the app in debug mode (without dummy hardware), add the flag -d to the above command::
    
    python doptical/app.py -d


Developing the GUI (Graphical User Interface)
----------------------------------------------
The GUI is designed using the Qt framework, using open-source Python Bindings (Pyside2).
A number of tools are available that facilitate development.
The Qt Designer tool can be found in the virtual environment directory, eg. venv\Lib\site-packages\PySide2\designer.exe

This graphical tool can be used to compose elements of the GUI visually. Designs are saved as .ui files. These must be compiled into Python files which can be imported into the application using the pyside2-uic tool. Eg::

    pyside2-uic --from-imports doptical\ui\app.ui -o doptical\ui\app.py

**NB. The --from-imports flag makes sure resources (eg. app_rc.py) are imported relative to their respective files (eg. app.py)**

To compile Qt resource files (which contain icon images, for example), use the pyside2-rcc executable which can be found in the site-packages folder along with the designer executable above. Eg::

    env\Lib\site-packages\Pyside2\pyside2-rcc.exe doptical\ui\app.qrc -o doptical\ui\app_rc.py

Source-code structure
---------------------
