# pseudospec -- a module for the pseudo-spectral method

## License
BSD

## Install

  python setup.py install --record files.txt

## Uninstall

For unix

    python setup.py install --record files.txt
    cat files.txt | xargs rm -rf

For Windows

    python setup.py install --record files.txt
    Get-Content files.txt | ForEach-Object {Remove-Item $_ -Recurse -Force}

## How to use

    Read files in the directory "samples".

## Notice

- Before updating, you might have to remove the egg file.
- One of code samples requires [Panel](https://panel.holoviz.org).


## Docker
At the project root, 

    $ docker-compose up

To stop and remove the container,

    $ docker-compose down
    
If you want to launch JupyterLab without token,

    $ docker-compose run --service-ports python start.sh jupyter lab --ports 8889 --LabApp.token=''

To install nose,

    $ docker exec -it <container name> start.sh
    $ conda install nose