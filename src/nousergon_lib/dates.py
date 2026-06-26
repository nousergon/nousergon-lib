"""Back-compat re-export — relocated to ``krepis.dates`` (MIT) in v0.66.0.

``nousergon_lib.dates`` is now an alias for :mod:`krepis.dates`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.dates`` directly.
"""

import sys

import krepis.dates as _mod

sys.modules[__name__] = _mod
