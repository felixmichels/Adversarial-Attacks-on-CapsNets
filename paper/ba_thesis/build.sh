#!/bin/sh

rm -f ba.aux ba.bbl ba.blg ba.lof ba.log ba.lot ba.out ba.pdf ba.run.xml ba.tox ba.-blx.bib

pdflatex ba.tex &&
bibtex ba.aux &&
pdflatex ba.tex &&
pdflatex ba.tex
