# SaltMix Beamer presentation

This directory contains the LaTeX Beamer presentation for the SaltMix manuscript:
`Physics-Inspired Neural Models for Mixture-Order Extrapolation of Molten Salt
Thermophysical and Transport Properties`.

It uses the Beamer class and tries the Metropolis theme first. If Metropolis is
not installed, it falls back to a stock Beamer theme so the file can still be
compiled on a minimal LaTeX setup.

## Compile

```bash
cd docs/saltmix-presentation
tectonic saltmix_presentation.tex
```

or:

```bash
cd docs/saltmix-presentation
latexmk -pdf saltmix_presentation.tex
```

## Notes

- Edit `saltmix_presentation.tex` directly.
- Put local slide figures in `docs/saltmix-presentation/figures/`.
- The presentation reuses figures from `docs/saltmix-manuscript/figures/`.
- Beamer documentation: <https://ctan.org/pkg/beamer>
- Metropolis theme: <https://ctan.org/tex-archive/macros/latex/contrib/beamer-contrib/themes/metropolis>
