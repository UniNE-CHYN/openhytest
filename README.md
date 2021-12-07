# openhytest
Package for well test analysis, inspired by the functionalities available in [hytool](https://github.com/UniNE-CHYN/hytool).

openhytest is currently structured into three modules:
- Pre-processing functions and utilities handling data preparation
- A library of flow models and analytic solutions, following an oriented-object programming structure
- Post-processing functions for results analysis, statistical summary and reporting.

List of models currently implemented:
- Theis, Radial flow (1935)
- Theis, Multirate test (1935)?
- Warren and Root, Double porosity (1936)
- Theis, No flow boundary (19??)
- Theis, Constant head boundary (1941)
- Hantush-Jacob, Leaky aquifer (1955)
- Jacob-Lohman, Constant head test (1952)
- Hvorslev (1957)
- Boulton, Delayed yield (1963)
- Papadopoulos and Cooper, Large diameter well (1967)
- Cooper, Bredehoeft and Papadopoulos, Shut-in pulse or slug test (1967) 
- Agarwal, Wellbore storage and skin (1970)
- Neuzil, Modified pulse test (1982)
- Barker, Generalized radial flow (1988)
