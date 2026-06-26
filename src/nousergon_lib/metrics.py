"""Back-compat re-export — relocated to ``krepis.metrics`` (MIT) in v0.66.0.

``nousergon_lib.metrics`` is now an alias for :mod:`krepis.metrics`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.metrics`` directly.
"""

import sys

import krepis.metrics as _mod

sys.modules[__name__] = _mod
