#!/bin/bash
HOSTNAME=$(hostname)
WORKER_TOKEN="SWMTKN-1-5zonedimczvleil8gns661lu3x53hwueg0lq8yec6wfb8k8m96-arv6d3f10zjnlb4ipkdi4p38s"
MANAGER_IP="192.168.1.64:2377"

# Overlay networks require ports:
#TCP port 2375 for Docker Engine access
#TCP port 2377 for cluster management communications
#TCP and UDP port 7946 for communication among nodes
#UDP port 4789 for overlay network traffic
#
NODENAME=swarm-node-"${HOSTNAME}"
echo "### Removing existing dind node.."
result=$(docker images -q "${NODENAME}" )
if [[ -n "$result" ]]; then
  echo "Container exists, stopping"
  docker rm $(docker stop "${NODENAME}")
else
  echo "No such container"
fi



echo "### Running Docker-in-Docker (DIND) image.."
docker run -d --privileged --name "${NODENAME}"\
    --hostname="${NODENAME}" \
    -p 12376:2376   \
    -p 2377:2377/tcp \
    -p 7946:7946 \
    -p 4789:4789/udp \
    docker:20.10.3-dind \
    dockerd --host=tcp://0.0.0.0:2376 --host=unix:///var/run/docker.sock

echo "### Waiting for DIND node to go up.."
sleep 30
docker --host tcp://localhost:12376 info
echo "### Joining DIND node to master.."
docker --host tcp://localhost:12376 \
  swarm join \
  --advertise-addr=192.168.1.102 \
  --token "${WORKER_TOKEN}" \
  "${MANAGER_IP}"

echo "### Done"