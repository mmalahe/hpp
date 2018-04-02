FROM ubuntu

WORKDIR /app

ADD . /app

RUN /app/install_deps_xenial.sh
