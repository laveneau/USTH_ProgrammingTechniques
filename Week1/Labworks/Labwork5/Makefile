# On macOS and most Linux distributions the command is python3; `python` often
# does not exist at all. Override it if yours is named differently:
#     make check PYTHON=python
PYTHON ?= python3

.PHONY: test part1 part2 part3 typecheck check clean

test:            ## run all the tests
	$(PYTHON) -m unittest discover -s tests -v

part1:           ## forward mode (dual numbers)
	$(PYTHON) -m unittest -v tests/test_dual.py

part2:           ## reverse mode (backpropagation)
	$(PYTHON) -m unittest -v tests/test_var.py

part3:           ## gradients of functions of several variables
	$(PYTHON) -m unittest -v tests/test_gradient.py

typecheck:       ## static typing
	$(PYTHON) -m mypy --strict autodiff

check: test typecheck   ## everything that must be green before you submit

clean:
	rm -rf __pycache__ */__pycache__ .mypy_cache
