"""Back-compat re-export — relocated to ``krepis.cost`` (MIT) in v0.66.0.

``nousergon_lib.cost`` is now an alias for :mod:`krepis.cost`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.cost`` directly.
"""

import sys

import krepis.cost as _mod

sys.modules[__name__] = _mod
