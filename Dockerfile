FROM ubuntu

WORKDIR /app

ADD . /app

RUN /app/scripts/install_deps_xenial.sh
