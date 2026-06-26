"""Back-compat re-export — relocated to ``krepis.alerts`` (MIT) in v0.66.0.

``nousergon_lib.alerts`` is now an alias for :mod:`krepis.alerts`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.alerts`` directly.
"""

import sys

import krepis.alerts as _mod

sys.modules[__name__] = _mod
