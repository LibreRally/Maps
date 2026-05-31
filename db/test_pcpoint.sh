#!/bin/sh
pg_ctlcluster 14 main start
sleep 2
psql -U postgres -c "CREATE EXTENSION pointcloud;"
psql -U postgres -c "SELECT PC_MakePoint(1, ARRAY[-127, 45, 124.0, 4.0]);"
