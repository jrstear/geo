QMDS := $(wildcard *.qmd)
PDFS := $(QMDS:.qmd=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.qmd
	quarto render $< --to pdf

clean:
	rm -f $(PDFS)
