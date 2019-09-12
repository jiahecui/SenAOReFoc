VENV := "venv"
UI_FILES := $(wildcard doptical/gui/ui/*.ui)
RC_FILES := $(wildcard doptical/gui/*.qrc)

COMPILED_UI := $(UI_FILES:%.ui=%.py)
COMPILED_RC := $(RC_FILES:%.qrc=%_rc.py)

UIC := $(shell find $(VENV) -type f -name 'pyside2-uic')
RCC := $(shell find $(VENV) -type f -name 'pyside2-rcc')

ifeq ($(OS),Windows_NT)
	OS_NAME := Windows
	IS_WINDOWS = 1
else
	OS_NAME := $(shell uname)
	IS_WINDOWS = 0
endif

#####################################

.PHONY: all run debug clean

all: ui rc

ui: ${COMPILED_UI}

rc: ${COMPILED_RC}

%.py: %.ui
	@echo "Compiling $< -> $@"
	@$(UIC)  --from-imports $< -o $@;

%_rc.py: %.qrc
	@echo "Compiling $< -> $@"
	@$(RCC) $< -o $@;

run:
	@if [ $(IS_WINDOWS) = 1 ]; then\
		$(VENV)\\Scripts\\activate; \
		python doptical\app.py; \
	else \
		source $(VENV)/bin/activate; \
		python doptical/app.py; \
	fi

debug:
	@if [ $(IS_WINDOWS) = 1 ]; then\
		$(VENV)\\Scripts\\activate; \
		python doptical\app.py; \
		echo "hello";\
	else \
		source $(VENV)/bin/activate; \
		python doptical/app.py -d; \
	fi

clean:
	@echo "Cleaning"
	rm -rvf ${COMPILED_UI} ${COMPILED_RC}
