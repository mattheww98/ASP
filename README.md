Absorption Spectrum Prediction model based on TLOpt and CrabNet

Label scaling is configured with the `ASP` constructor's `label_scaling`
argument. Supported values are `"z"` (default), `"log"`, `"minmax"` (or
`"min-max"`), and `"median"`. For example, add `"label_scaling": "median"` to
`config.json` to divide labels by the training-set median of spectrum maxima.
