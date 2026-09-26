# Installing Nimbus Queue

## Requirements

Nimbus Queue needs a 64-bit Linux host with at least 2 GB of RAM. Windows and macOS are supported for development only, not for production.

## Install steps

Download the release archive, unpack it, and run `nimbus init` to create a default configuration. The configuration file is written to `/etc/nimbus/nimbus.toml`.

Start the broker with `nimbus start`. Check that it is running with `nimbus status`, which prints the node role and the number of owned partitions.

## Upgrading

Upgrades are rolling: upgrade follower nodes one at a time, then the leader last. Never skip more than one minor version in a single upgrade.
