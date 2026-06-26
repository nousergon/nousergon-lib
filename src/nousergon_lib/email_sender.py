"""Back-compat re-export — relocated to ``krepis.email_sender`` (MIT) in v0.66.0.

``nousergon_lib.email_sender`` is now an alias for :mod:`krepis.email_sender`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.email_sender`` directly.
"""

import sys

import krepis.email_sender as _mod

sys.modules[__name__] = _mod
