# Copyright Ryo Ikota
ARG BASE_CONTAINER=jupyter/scipy-notebook:7a0c7325e470
FROM $BASE_CONTAINER

ENV PKGDIR=$HOME/package

COPY --chown=$NB_USER:$NB_GID setup.py $PKGDIR/ 
COPY --chown=$NB_USER:$NB_GID pseudospec/ $PKGDIR/pseudospec 
USER $NB_UID
RUN cd $PKGDIR ; python setup.py install --record files.txt
WORKDIR $HOME/work
CMD ["start-notebook.sh"]
