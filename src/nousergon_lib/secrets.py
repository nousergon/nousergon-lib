"""Back-compat re-export — relocated to ``krepis.secrets`` (MIT) in v0.66.0.

``nousergon_lib.secrets`` is now an alias for :mod:`krepis.secrets`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.secrets`` directly.
"""

import sys

import krepis.secrets as _mod

sys.modules[__name__] = _mod
