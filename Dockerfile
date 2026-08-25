FROM alpine:edge

RUN apk add --no-cache \
    clang \
    clang-extra-tools \
    cmake \
    build-base \
    ninja \
    zsh \
    shadow

RUN chsh -s /bin/zsh root
