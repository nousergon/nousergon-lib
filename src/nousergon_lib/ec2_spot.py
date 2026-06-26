"""Back-compat re-export — relocated to ``krepis.ec2_spot`` (MIT) in v0.66.0.

``nousergon_lib.ec2_spot`` is now an alias for :mod:`krepis.ec2_spot`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.ec2_spot`` directly.
"""

import sys

import krepis.ec2_spot as _mod

sys.modules[__name__] = _mod
