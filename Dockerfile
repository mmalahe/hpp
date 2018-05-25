FROM ubuntu

WORKDIR /app

ADD . /app

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive apt install --yes ssh

RUN /app/scripts/install_deps_xenial.sh
