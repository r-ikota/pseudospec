# Copyright Ryo Ikota
ARG BASE_CONTAINER=jupyter/scipy-notebook:python-3.9.13
FROM $BASE_CONTAINER

USER $NB_UID
COPY requirements-conda.txt /tmp/
RUN mamba install --yes --file /tmp/requirements-conda.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

ENV PKGDIR=$HOME/package

COPY --chown=$NB_USER:$NB_GID setup.py $PKGDIR/
COPY --chown=$NB_USER:$NB_GID setup.cfg $PKGDIR/ 
COPY --chown=$NB_USER:$NB_GID src/pseudospec/ $PKGDIR/src/pseudospec 
COPY --chown=$NB_USER:$NB_GID tests/ $HOME/tests 
USER $NB_UID
RUN cd $PKGDIR ; python setup.py install --record files.txt

CMD ["start.sh", "jupyter", "lab", "--port=8899"]
