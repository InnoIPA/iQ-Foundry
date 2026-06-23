# Build the iQF Docker Image

This guide explains how to build the `iQF` Docker image from the repository source.

> [!IMPORTANT]
> Pulling the published image from Docker Hub is the recommended workflow for standard usage. Local
> image builds are intended as a fallback path when you need to rebuild the image from source.

The recommended build entrypoint is:

```bash
./docker/iqf build
```

This uses the Dockerfile at [docker/Dockerfile](./Dockerfile) and builds the image from the repo
root.

## Prerequisites

Before building, make sure:

- Docker Engine is installed and working
- you are running commands from the repository root
- the repository contains `docker/Dockerfile` and `requirements/host.txt`

## Recommended Build Flow

Build the default image:

```bash
./docker/iqf build
```

The default image tag is:

```text
iq-foundry:local
```

To use this built image with `./docker/iqf run ...` and `./docker/iqf shell`, export the image
name in each new terminal:

```bash
export IQF_DOCKER_IMAGE=iq-foundry:local
```

Otherwise the wrapper will fall back to the published default image and Docker may pull it.

Use dry-run to inspect the generated Docker command without building:

```bash
./docker/iqf build --dry-run
```

`./docker/iqf run ...` and `./docker/iqf shell` still default to the published image unless you
override them with `--image` or `IQF_DOCKER_IMAGE`.

## Build with a Custom Tag

Override the output image tag with `--image`:

```bash
./docker/iqf build --image my-iqf:dev
```

You can also use the environment variable:

```bash
IQF_DOCKER_IMAGE=my-iqf:dev ./docker/iqf build
```

## Direct Docker Equivalent

If you need the raw Docker command, the wrapper build is equivalent to:

```bash
docker build -f docker/Dockerfile -t iq-foundry:local .
```

Run that from the repository root.

## Verify the Image

After the build completes, you can start a shell in the image:

```bash
./docker/iqf shell --image iq-foundry:local
```

Or verify directly with Docker:

```bash
docker run --rm -it iq-foundry:local bash
```

## Next Step

After the image is built, continue with the host workflow guides:

- [Ubuntu_host.md](../Ubuntu_host.md)
- [Windows_host.md](../Windows_host.md)

For mode-specific usage and advanced flags, see:

- [docs/qc_mode.md](../docs/qc_mode.md)
- [docs/mAP_mode.md](../docs/mAP_mode.md)
- [docs/test_mode.md](../docs/test_mode.md)
