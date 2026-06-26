"""Back-compat re-export — relocated to ``krepis.ssm_log_capture`` (MIT) in v0.66.0.

``nousergon_lib.ssm_log_capture`` is now an alias for :mod:`krepis.ssm_log_capture`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.ssm_log_capture`` directly.
"""

import sys

import krepis.ssm_log_capture as _mod

sys.modules[__name__] = _mod
