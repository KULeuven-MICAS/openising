# Docker container

This directory contains the [Docker file](Dockerfile) used to build the `ising` Docker container. The container is based on the Ubuntu 24.04 LTS image and comes with all free development tools. The environment is also already configured, such that no additional steps are required to work in the container after installation.

## Installation

### Pre-built container

There is a pre-built version of the container available online. This version is up to date with the latest developments on the `main` branch. The CI publishes a new container every time a new commit is pushed to this branch.

To download the container, first login to the GitHub container registry:
```shell
$ echo "<your token>" | podman login ghcr.io -u <your-github-username> --password-stdin
```

As a password you should use a
[personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
that at least has package registry read permission.

You can then install the container by running:
```shell
$ podman pull ghcr.io/kuleuven-micas/ising:main
```

### Build instructions

In case you cannot use the pre-built container, e.g. if you need to make changes to the Dockerfile, you can build the
container locally by running the following command in the root of the repository:

```shell
$ podman build -t ghcr.io/kuleuven-micas/ising:main -f util/container/Dockerfile .
```
During the building process, it will read the requirements.txt.

Than you can upload the image to the github
```shell
$ podman pull -t ghcr.io/kuleuven-micas/ising:main
```
## Usage

To run the container in interactive mode:

```shell
$ podman run -it --rm -v $REPO_TOP:/repo -w /repo ghcr.io/kuleuven-micas/ising:main
```

or you could write a .sh file to run

Suppose your dir structure is
```
/your-repo
    /ising
    start_ising.sh
```
You could write a start_ising.sh file as
```
podman run --rm --workdir /your-gitrepo -it -v /your-gitrepo:/your-gitrepo ghcr.io/kuleuven-micas/ising:main
```
Than you could run 
```shell
$ source start_ising.sh
```
under /your-repo.