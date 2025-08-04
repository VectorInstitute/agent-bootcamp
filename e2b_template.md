# Custom E2B Template

The sandboxes are non-persistent. Each time your agent runs a command, all data files need to be uploaded again. If your data files are large (10MB total or more), it might take minutes before any command starts running. For performance reasons, you should consider bundling these data files into a custom E2B template. Here are the quick steps for setting that up.

Estimated time: 30 minutes.

## Prerequisites

Install Docker Engine ([Docker CE](https://docs.docker.com/engine/install/)).

Install Node Version Manager ([NVM](https://github.com/nvm-sh/nvm)).

Install the latest LTS Node version (22 as of 2025AUG03.)

## Steps

Modify `e2b.Dockerfile`. Replace the URL with the link to your data files.
Important, try not to touch the `/home/user` folder, as that might create permission issues and break the Python installation.


## Testing Locally

Try building the image locally.

```bash
docker build .  -f e2b.Dockerfile  --tag e2b-example
```

If the build works, run the following to enter the shell environment within the container. Make sure you can find your files under `/data`, as that is where your agent would access those data files.

```bash
docker run -it --entrypoint /bin/bash e2b-example

# Within the container
ls -lh /data

# You should see your data file listed.
```

## Push to E2B

If the local tests looks reasonable, push the image to E2B as a template. You might need to login to your E2B account.

```bash
# The command "/root/.jupyter/start-up.sh" is from the E2B base image
# If you are only adding data files, you don't need to modify this line.

e2b template build -c "/root/.jupyter/start-up.sh"
```

If the build is finished properly, you should see output like the following:

> ✅ Building sandbox template 9p6favrrqijhasgkq1tv finished.

## Modify your "System Prompt"

Previously, the agent assumes that all data files are under the initial working directory. That is not the case here. You should modify the system prompt and instruct the agent to look under `/data` (or whatever folder you specified in the Dockerfile.)
