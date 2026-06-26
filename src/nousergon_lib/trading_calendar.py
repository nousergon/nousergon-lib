"""Back-compat re-export — relocated to ``krepis.trading_calendar`` (MIT) in v0.66.0.

``nousergon_lib.trading_calendar`` is now an alias for :mod:`krepis.trading_calendar`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.trading_calendar`` directly.
"""

import sys

import krepis.trading_calendar as _mod

sys.modules[__name__] = _mod
