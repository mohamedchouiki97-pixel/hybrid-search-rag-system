# Nimbus Queue Overview

Nimbus Queue is a lightweight message broker for small teams. It was first released in 2019 by the Halden Labs engineering group.

## Architecture

A Nimbus cluster is made of one leader node and up to six follower nodes. Messages are grouped into partitions, and each partition is owned by exactly one node at a time through a partition lease.

Messages are written to append-only segment files. Each segment file holds at most 128 MiB of messages before a new segment is started.

## Networking

The broker listens on TCP port 7420 by default. The admin dashboard is served on port 7421. All traffic between nodes is encrypted with TLS 1.3.
