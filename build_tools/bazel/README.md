### Note regarding this folder

Currently, these scripts are also being called from our roof-mlir repository. Therefore, it is important to prefix targets with @patio_cellar_hexagon and not used naked //... here.
In the future, it is likely that this will have to be reworked if we want an independent build system. In particular, the mapping of the @targets might need to be restructured.
