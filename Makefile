IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR)

portainer-up:
	docker stack deploy -c portainer-agent-stack.yml --orchestrator swarm portainer

portainer-down:
	docker stack rm portainer

api-up:
	docker stack deploy -c docker-swarm.yml --orchestrator swarm cryptoml

api-down:
	docker stack rm cryptoml

login:
	docker login

image:
	DOCKER_BUILDKIT=1 \
	docker build -t mrlicorice/cryptoml-api:latest .

push: image
	docker push mrlicorice/cryptoml-api:latest


