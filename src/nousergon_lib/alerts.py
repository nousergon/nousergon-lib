"""Back-compat re-export — relocated to ``krepis.alerts`` (MIT) in v0.66.0.

``nousergon_lib.alerts`` is now an alias for :mod:`krepis.alerts`. Importing it
rebinds this module object to the krepis one, so the full public surface
(including private helpers such as flow-doctor secret seeding) resolves
unchanged. New code should import from ``krepis.alerts`` directly.
"""

import sys

import krepis.alerts as _mod

sys.modules[__name__] = _mod

# `python -m nousergon_lib.<name>` must DELEGATE to the krepis CLI, not fall
# off the end of this shim with exit 0. Without this guard, runpy executes the
# shim as __main__ (the target's own guard sees __name__ ==
# "krepis.<name>" and never fires), so the invocation is a silent no-op —
# the 2026-07-03 weekly SF ran ZERO EC2 workloads while reporting SUCCESS
# because every spot command was wrapped in exactly that no-op (config#1646).
if __name__ == "__main__":
    sys.exit(_mod.main())
