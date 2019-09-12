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

There are many text editors/IDEs available. A recommended free editor if you're unsure is Microsoft Visual Studio Code:
https://code.visualstudio.com/

Getting started
---------------
Once you have installed the tools above, clone the source-code from the Git repository::

    git clone https://github.com/jiahecui/SensorbasedAO.git

Create a virtual environment for development (Optional but recommended)::
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the venv module to create an isolated virtual environment. The required packages are installed using pip. Finally, the main source is installed using pip in an editable format. Changes to the source affect the application the next time it is run.
Run the following from the folder the top folder (that includes setup.py)::

	python -m venv venv

Activate virtual environment before next steps:
    
Windows:

	venv\Scripts\activate

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

How to start a new git repository
---------------------
A new repo from scratch:

    Create a directory to contain the project.
    Go into the new directory.
    Type git init.
    Write some code.
    Type git add <file> to add the files.
    Type git commit -m <message>.

Connect it to github:

    Go to github.
    Log in to your account.
    Click the new repository button in the top-right. You’ll have an option there to initialize the repository with a README file, but I don’t.
    Click the “Create repository” button.

Now, follow the second set of instructions, “Push an existing repository…”

    git remote add origin https://github.com/username/new_repo
    git push -u origin master

How to create your own branch in remote repository
--------------------------------------------------
Create a new branch called <branch>. This does not check out the new branch.

    git branch <branch>

Checkout the new branch.

    git checkout <branch_name>

Push the new branch onto the remote repository.

    git push -u origin <branch_name>

You can verify on the Github web browser that your branch is present.

