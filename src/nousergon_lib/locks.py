"""Back-compat re-export — relocated to ``krepis.locks`` (MIT) in v0.66.0.

``nousergon_lib.locks`` is now an alias for :mod:`krepis.locks`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.locks`` directly.
"""

import sys

import krepis.locks as _mod

sys.modules[__name__] = _mod
