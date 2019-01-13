# pseudospec -- a module for the pseudo-spectral method

## License
BSD

## Install
  python setup.py Install

## Uninstall
For unix

    python setup.py install --record files.txt
    cat files.txt | xargs rm -rf

For Windows

    python setup.py install --record files.txt
    Get-Content files.txt | ForEach-Object {Remove-Item $_ -Recurse -Force}

## How to use

    Read files in the directory "samples".



 