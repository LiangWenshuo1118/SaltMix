# Beamer presentation template

This is a lightweight LaTeX template for research presentations.

It uses the Beamer class and tries the Metropolis theme first. If Metropolis is
not installed, it falls back to a stock Beamer theme so the file can still be
compiled on a minimal LaTeX setup.

## Compile

```bash
cd docs/beamer-presentation-template
tectonic main.tex
```

or:

```bash
cd docs/beamer-presentation-template
latexmk -pdf main.tex
```

## Notes

- Edit `main.tex` directly.
- Put local slide figures in `docs/beamer-presentation-template/figures/`.
- The template can also reuse figures from `docs/jctc-latex-template/figures/`.
- Beamer documentation: <https://ctan.org/pkg/beamer>
- Metropolis theme: <https://ctan.org/tex-archive/macros/latex/contrib/beamer-contrib/themes/metropolis>
