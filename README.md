# pseudospec -- a module for the pseudo-spectral method

## License
BSD

## Install
  python setup.py install

## Uninstall
For unix

    python setup.py install --record files.txt
    cat files.txt | xargs rm -rf

For Windows

    python setup.py install --record files.txt
    Get-Content files.txt | ForEach-Object {Remove-Item $_ -Recurse -Force}

## How to use

    Read files in the directory "samples".



## Docker
At the project root, 

    $ docker-compose up

To stop and remove the container,

    $ docker-compose down
    