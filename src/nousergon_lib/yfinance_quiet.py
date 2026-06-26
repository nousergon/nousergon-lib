"""Back-compat re-export — relocated to ``krepis.yfinance_quiet`` (MIT) in v0.67.0.

``nousergon_lib.yfinance_quiet`` is now an alias for
:mod:`krepis.yfinance_quiet`. Importing it rebinds this module object to the
krepis one, so the full public surface resolves unchanged. New code should
import from ``krepis.yfinance_quiet`` directly.
"""

import sys

import krepis.yfinance_quiet as _mod

sys.modules[__name__] = _mod
