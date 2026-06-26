"""Back-compat re-export — relocated to ``krepis.http_retry`` (MIT) in v0.66.0.

``nousergon_lib.http_retry`` is now an alias for :mod:`krepis.http_retry`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.http_retry`` directly.
"""

import sys

import krepis.http_retry as _mod

sys.modules[__name__] = _mod
