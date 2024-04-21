# Docker setup

This docker setup is tested on WSL(Ubuntu).

make sure you are under directory yourworkspace/instantmesh/

run

`docker build -t instantmesh/deploy:cuda12.1 -f docker/Dockerfile .`

then run

`docker run --gpus all -it instantmesh/deploy:cuda12.1`