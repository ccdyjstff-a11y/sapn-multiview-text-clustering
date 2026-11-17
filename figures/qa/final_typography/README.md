# Four supplementary figure typography revisions

The four PDFs are ready for integration. They retain every plotted scientific coordinate, the original plot rectangles and ranges, colours, markers, eligibility decisions and sample-standard-deviation definition. Body text is now at least 8.16 pt when placed at 166 mm width. In the numerical plot only, mathematical superscripts/subscripts are about 5.88 pt. The reference curve and stopping threshold have not moved.

The old/new rendered PDFs were parsed independently with PyMuPDF. All 849 coloured scientific paths inside the original axes match at 0.001 pt precision, including markers, trajectories and error-bar paths. No scientific path was added or removed. Full details are in `FONT_POLISH_QA.json`.

All four figures pass new text, collision and strict panel-alignment checks. Every final panel was visually inspected. The larger Reuters legend needed extra vertical separation, and the bottom-row mechanism titles needed a spacing adjustment. Both are resolved. The generic source audit has 20 passes and one nonblocking warning for no TIFF. The vector PDF is authoritative; the 600 dpi PNG is a preview and SVG retains text.

Reproduce using the existing public repository's data and study-authored alignment helper:

```powershell
python plot_final_figure_fonts.py --release-root PATH_TO_PUBLIC_REPOSITORY --output OUTPUT_DIRECTORY
```

The script reads four public source-data files, recorded with their SHA256 hashes in `FONT_POLISH_PROVENANCE.json`. It needs NumPy and Matplotlib. It performs no fitting, label scoring or model selection and requires no private research paths. PDF serialization dates can change on rerender; the source values and geometry remain verifiable.

The other unchanged figures were inspected separately. Several dense panels have 7.38 pt body text at 166 mm; those are readable, embedded vector text and have no clipping or collisions. The 5.16 pt minimum in the model schematic is a mathematical subscript, not an axis or legend font. This package does not describe every manuscript figure as having 8 pt body text.
