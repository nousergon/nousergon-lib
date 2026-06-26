"""Back-compat re-export — relocated to ``krepis.logging`` (MIT) in v0.66.0.

``nousergon_lib.logging`` is now an alias for :mod:`krepis.logging`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.logging`` directly.
"""

import sys

import krepis.logging as _mod

sys.modules[__name__] = _mod
