IDIR =../include
CC=gcc
REQUIREMENTS= app/cryptoml/requirements.txt app/cryptoml_api/requirements.txt app/cryptoml_core/requirements.txt

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

requirements: $(REQUIREMENTS)
	cat $(REQUIREMENTS) > requirements.txt

install: requirements
	pip install -r requirements.txt

net-up:
	docker network create -d overlay --attachable cryptoml_internal

net-down:
	docker network rm cryptoml_internal


