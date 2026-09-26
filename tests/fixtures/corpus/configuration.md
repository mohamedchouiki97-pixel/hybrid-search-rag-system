# Configuring Nimbus Queue

All settings live in `nimbus.toml`. Changes take effect after `nimbus reload`.

## Storage settings

`data_dir` sets where segment files are stored. The default is `/var/lib/nimbus`.

`retention_hours` sets how long messages are kept before deletion. The default is 72 hours.

## Message settings

`max_message_size` limits the size of a single message. The default is 4 MiB and the hard upper limit is 64 MiB.

## Lease settings

`lease_timeout_seconds` sets how long a node can go without a heartbeat before it loses its partition leases. The default is 15 seconds.
