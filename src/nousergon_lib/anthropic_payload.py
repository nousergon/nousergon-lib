"""Back-compat re-export — relocated to ``krepis.anthropic_payload`` (MIT) in v0.66.0.

``nousergon_lib.anthropic_payload`` is now an alias for :mod:`krepis.anthropic_payload`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.anthropic_payload`` directly.
"""

import sys

import krepis.anthropic_payload as _mod

sys.modules[__name__] = _mod
