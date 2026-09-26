# Operating Nimbus Queue

## Backups

Run `nimbus backup --to <path>` to copy all closed segment files to another location. Open segments are skipped. Backups are incremental: only segments created since the last backup are copied.

## Monitoring

Nimbus exposes Prometheus metrics at `/metrics` on the admin dashboard port. The most important metric is `nimbus_lease_renewals_failed_total`, which rises before lease errors appear.

## Scaling

To add capacity, add follower nodes up to the cluster limit. Partitions are rebalanced automatically within ten minutes of a node joining.
