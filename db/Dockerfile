FROM pgpointcloud/pointcloud:latest

# Install PostGIS (may trigger upgrades but that's fine — PC_MakePoint is broken upstream)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-14-postgis-3 \
        postgresql-14-postgis-3-scripts \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Known issue: pgpointcloud 1.2.5-3 PC_MakePoint has a memory alloc bug
# (confirmed in both fresh image and custom build).
# Point clouds stored as PostGIS geometry tables; pcpatch deferred to fix.
RUN echo "pgpointcloud + PostGIS ready. PC_MakePoint awaiting upstream fix."
