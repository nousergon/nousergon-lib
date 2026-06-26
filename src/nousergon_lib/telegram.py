"""Back-compat re-export — relocated to ``krepis.telegram`` (MIT) in v0.66.0.

``nousergon_lib.telegram`` is now an alias for :mod:`krepis.telegram`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.telegram`` directly.
"""

import sys

import krepis.telegram as _mod

sys.modules[__name__] = _mod
