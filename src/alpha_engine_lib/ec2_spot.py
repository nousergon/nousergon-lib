"""
EC2 spot-launch capacity-resilience chokepoint.

Consolidation substrate for the spot-launch pattern that previously
appeared as three mirrored copies of the same fragility across the
alpha-engine fleet — each repo's launcher script (``spot_data_weekly.sh``
in alpha-engine-data, ``spot_train.sh`` in alpha-engine-predictor,
``spot_backtest.sh`` in alpha-engine-backtester) independently encoded
the same hardcoded ``--subnet-id`` (single AZ, us-east-1f) +
``--instance-type c5.large`` (single SKU) + N retries-with-backoff. When
AWS ran out of c5.large capacity in us-east-1f, every spot-launching
state failed simultaneously with no resilience.

**Why now (2026-05-22 evening):** The post-trap-fix dry-pass of the
Saturday SF (``postfix-keystone-20260522T232655Z``) hit
``InsufficientInstanceCapacity`` on the Evaluator's spot launch in
us-east-1f. The 2 earlier spots (Backtester + Parity) happened to clear
because AWS capacity rolled between the launches; Evaluator drew the
short straw. The defect class is "any single Saturday SF run has a
non-trivial chance of hitting capacity in at least one of the 3+ spot
states." The Friday-PM dry-pass exposed it (third in a row caught break
of the dry-pass safety net — first was the trap escape, second was the
keystone merge order, third is this).

**Why a CLI, not a bash function:**

Per ``~/Development/CLAUDE.md`` SOTA sub-sub-rule — "when mirroring a
pattern across repos, consider lifting it into ``alpha-engine-lib``...
Pure-Bash primitives can stay mirrored unless re-expressible as a Python
CLI entry callable from Bash, in which case the CLI re-expression is
the institutional path." Third repo with the same fragility is well
past the second-recurrence trigger. The CLI shape mirrors
:mod:`alpha_engine_lib.alerts` + :mod:`alpha_engine_lib.ssm_log_capture`
precedent.

**Strategy:**

The function iterates ``(instance_type, subnet)`` combinations in the
order given, attempting :func:`RunInstances` against each. On
``InsufficientInstanceCapacity`` / ``InsufficientHostCapacity`` /
``Unsupported`` (instance type not in AZ) → rotate to the next
combination. On any other error (auth, quota, AMI not found) → raise.

Caller controls the rotation order by listing types/subnets. Default
shape we use in the fleet:

- types: ``[c5.large, m5.large, c6i.large, c5a.large]`` (all 2 vCPU /
  ~4-8 GB RAM; capacity-resilient set chosen 2026-05-22)
- subnets: all default-VPC subnets across us-east-1{a,b,c,d,e,f}

**Public API:**

- :func:`launch` — Python API returning ``InstanceId`` on success,
  raising :class:`SpotCapacityExhausted` if every combination hit a
  capacity error, or :class:`SpotLaunchError` on any other error.
- CLI: ``python -m alpha_engine_lib.ec2_spot launch --types ... --subnets ...``.
  Returns ``InstanceId`` on stdout. Exits non-zero on failure;
  capacity-exhaustion exits 64 (distinguishable from generic failure).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Final, Sequence

logger = logging.getLogger(__name__)

# Error codes (RunInstances) that mean "this combination is out of
# capacity, try another." Anything else is a hard error.
CAPACITY_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "InsufficientInstanceCapacity",
        "InsufficientHostCapacity",
        "SpotMaxPriceTooLow",  # spot-specific; AWS returns this when
                                # the AZ's spot price exceeds bid
        "Unsupported",          # instance type not offered in AZ
        "InvalidAvailabilityZone",
    }
)

CAPACITY_EXIT_CODE: Final[int] = 64


class SpotLaunchError(Exception):
    """Non-capacity RunInstances failure (auth, quota, AMI not found, …)."""


class SpotCapacityExhausted(SpotLaunchError):
    """Every (instance_type, subnet) combination returned a capacity error."""


def _build_run_instances_kwargs(
    *,
    image_id: str,
    instance_type: str,
    key_name: str,
    security_group_ids: list[str],
    subnet_id: str,
    iam_instance_profile: str,
    spot: bool,
    volume_size_gb: int,
    volume_type: str,
    shutdown_behavior: str,
    tag_name: str | None,
) -> dict:
    kwargs: dict = {
        "ImageId": image_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "SecurityGroupIds": security_group_ids,
        "SubnetId": subnet_id,
        "IamInstanceProfile": {"Name": iam_instance_profile},
        "MinCount": 1,
        "MaxCount": 1,
        "InstanceInitiatedShutdownBehavior": shutdown_behavior,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": volume_size_gb, "VolumeType": volume_type},
            }
        ],
    }
    if spot:
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    if tag_name:
        kwargs["TagSpecifications"] = [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": tag_name}],
            }
        ]
    return kwargs


def launch(
    instance_types: Sequence[str],
    subnets: Sequence[str],
    *,
    image_id: str,
    key_name: str,
    security_group_ids: Sequence[str],
    iam_instance_profile: str,
    spot: bool = True,
    volume_size_gb: int = 30,
    volume_type: str = "gp3",
    shutdown_behavior: str = "terminate",
    tag_name: str | None = None,
    region: str = "us-east-1",
) -> str:
    """Launch a spot, rotating across instance_types × subnets on capacity error.

    Returns:
        Instance ID of the first successful launch.

    Raises:
        SpotCapacityExhausted: every (type, subnet) combination returned
            a capacity error. Caller can wait + retry, or escalate.
        SpotLaunchError: any other RunInstances error (auth, quota,
            AMI not found, …) — these don't retry, they raise loud.
        ValueError: empty instance_types or subnets list.
    """
    if not instance_types:
        raise ValueError("instance_types must be non-empty")
    if not subnets:
        raise ValueError("subnets must be non-empty")

    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=region)
    sg_ids = list(security_group_ids)

    capacity_attempts: list[str] = []
    for instance_type in instance_types:
        for subnet_id in subnets:
            kwargs = _build_run_instances_kwargs(
                image_id=image_id,
                instance_type=instance_type,
                key_name=key_name,
                security_group_ids=sg_ids,
                subnet_id=subnet_id,
                iam_instance_profile=iam_instance_profile,
                spot=spot,
                volume_size_gb=volume_size_gb,
                volume_type=volume_type,
                shutdown_behavior=shutdown_behavior,
                tag_name=tag_name,
            )
            try:
                resp = ec2.run_instances(**kwargs)
            except ClientError as exc:
                err = exc.response.get("Error", {})
                code = err.get("Code", "UnknownError")
                msg = err.get("Message", str(exc))
                if code in CAPACITY_ERROR_CODES:
                    capacity_attempts.append(f"{instance_type}@{subnet_id}: {code}")
                    logger.warning(
                        "ec2_spot: %s in %s for %s — rotating",
                        code,
                        subnet_id,
                        instance_type,
                    )
                    print(
                        f"ec2_spot: {code} for {instance_type}@{subnet_id} — rotating",
                        file=sys.stderr,
                    )
                    continue
                raise SpotLaunchError(
                    f"RunInstances failed with non-capacity error "
                    f"{code} ({instance_type}@{subnet_id}): {msg}"
                ) from exc

            instance_id = resp["Instances"][0]["InstanceId"]
            logger.info(
                "ec2_spot: launched %s as %s in %s",
                instance_type,
                instance_id,
                subnet_id,
            )
            print(
                f"ec2_spot: launched {instance_type} as {instance_id} in {subnet_id}",
                file=sys.stderr,
            )
            return instance_id

    raise SpotCapacityExhausted(
        f"every (instance_type, subnet) combination returned a capacity error "
        f"({len(capacity_attempts)} attempts): "
        + "; ".join(capacity_attempts)
    )


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_engine_lib.ec2_spot",
        description=(
            "Launch an EC2 spot with capacity-resilient rotation across "
            "instance types and subnets. The institutional replacement for "
            "the hardcoded single-subnet + single-instance-type pattern "
            "mirrored across the alpha-engine fleet's spot launchers."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch_p = subparsers.add_parser(
        "launch",
        help="Launch a spot with rotating (type, subnet) combinations.",
    )
    launch_p.add_argument(
        "--types",
        required=True,
        help=(
            "Comma-separated instance types to try in order "
            "(e.g., 'c5.large,m5.large,c6i.large'). First success wins."
        ),
    )
    launch_p.add_argument(
        "--subnets",
        required=True,
        help=(
            "Comma-separated subnet IDs to try in order. Each is an AZ "
            "(default-VPC subnets in us-east-1 span 1a-1f)."
        ),
    )
    launch_p.add_argument("--image-id", required=True, help="AMI ID.")
    launch_p.add_argument("--key-name", required=True, help="EC2 key pair name.")
    launch_p.add_argument(
        "--security-group",
        required=True,
        action="append",
        help=(
            "Security group ID. Pass multiple times for >1 SG: "
            "--security-group sg-A --security-group sg-B"
        ),
    )
    launch_p.add_argument(
        "--iam-profile",
        required=True,
        help="IAM instance profile NAME (not ARN).",
    )
    launch_p.add_argument(
        "--no-spot",
        action="store_true",
        help="Launch on-demand instead of spot.",
    )
    launch_p.add_argument(
        "--volume-size",
        type=int,
        default=30,
        help="Root EBS volume size in GB (default: 30).",
    )
    launch_p.add_argument(
        "--volume-type",
        default="gp3",
        help="Root EBS volume type (default: gp3).",
    )
    launch_p.add_argument(
        "--shutdown-behavior",
        default="terminate",
        choices=("terminate", "stop"),
        help="Instance-initiated shutdown behavior (default: terminate).",
    )
    launch_p.add_argument(
        "--name",
        default=None,
        help="Name tag applied to the launched instance.",
    )
    launch_p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: $AWS_REGION or us-east-1).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    try:
        instance_id = launch(
            instance_types=_split_csv(args.types),
            subnets=_split_csv(args.subnets),
            image_id=args.image_id,
            key_name=args.key_name,
            security_group_ids=args.security_group,
            iam_instance_profile=args.iam_profile,
            spot=not args.no_spot,
            volume_size_gb=args.volume_size,
            volume_type=args.volume_type,
            shutdown_behavior=args.shutdown_behavior,
            tag_name=args.name,
            region=args.region,
        )
    except SpotCapacityExhausted as exc:
        print(f"ec2_spot: capacity exhausted: {exc}", file=sys.stderr)
        return CAPACITY_EXIT_CODE
    except SpotLaunchError as exc:
        print(f"ec2_spot: launch failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ec2_spot: bad input: {exc}", file=sys.stderr)
        return 2

    # InstanceId on stdout — bash callers capture this via $(...)
    print(instance_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
