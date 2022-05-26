FROM pytorch/pytorch:latest
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000
RUN apt-get -qy update \
 && apt-get -qy install --no-install-recommends curl \
 && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
 && apt-get -qy install --no-install-recommends git build-essential git-lfs \
 && conda install -c conda-forge jupyterlab ipywidgets nodejs=16.6.1 \
 && git lfs install
COPY setup.py setup.cfg /arqmath3/
WORKDIR /arqmath3
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
ENV PATH="/home/$UNAME/.local/bin:${PATH}"
RUN pip install .[all,notebook]
