# pseudospec -- a module for the pseudo-spectral method

## License
BSD

## Install

Type the following at the project root:

```
pip install .
```


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