# Nimbus Queue Error Codes

## ERR-7Q42

Error ERR-7Q42 means a partition lease expired while a write was in progress. The write is rejected and the client should retry. Frequent ERR-7Q42 errors usually mean the lease timeout is too low for the network.

## ERR-3B10

Error ERR-3B10 means a message exceeded the configured maximum message size. Split the payload or raise the limit.

## ERR-9Z05

Error ERR-9Z05 means the data directory is full. Free disk space or lower the retention period.
