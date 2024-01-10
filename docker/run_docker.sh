#! /bin/bash

docker run --rm -it \
	--net=host \
	--gpus all \
	-v $PWD/../:/tfmatch:rw \
	-v /media/gidobot/data:/media/gidobot/data:rw \
	-v ~/.docker_bash_history:/root/.bash_history \
	-w /tfmatch \
	docker:tfmatch \
	bash

# -v /media/kraft/af7cd17b-9563-477c-be1d-89aa6b8aebb6/data:/data:rw \
	# --ipc=host \
	# --env=NVIDIA_DRIVER_CAPABILITIES=all \
	# --env=DISPLAY \
	# --env=QT_X11_NO_MITSHM=1 \
	# --pid=host \
	# --cap-add=SYS_ADMIN \
	# --cap-add=SYS_PTRACE \
	# -v /tmp/.X11-unix:/tmp/.X11-unix:rw \


	# -u $(id -u):$(id -g) \