.PHONY: help html pdf clean

help:
	@echo "Targets:"
	@echo "  make html   – Build HTML docs"
	@echo "  make pdf    – Build PDF docs (requires LaTeX)"
	@echo "  make clean  – Remove build artifacts"

html:
	sphinx-build -b html docs _build/html

pdf:
	sphinx-build -b latex docs _build/latex
	cd _build/latex && pdflatex *.tex && pdflatex *.tex

clean:
	rm -rf _build
