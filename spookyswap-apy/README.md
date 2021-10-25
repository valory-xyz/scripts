Description
===========
The `apy.py` script accepts two positional arguments, the token ids of a SpookySwap pool's pairs, and outputs the pool's
APY.
> 📝 The token ids must be given in the correct order.

Run
===
This demo uses [poetry](https://python-poetry.org/) for python packaging and dependency management.

Example
-------

1. Example run for USDC-WFTM:
    ```shell
    python apy.py 0x04068da6c83afcfa0e13ba15a6696662335d5b75 0x21be370d5312f44cb42ce377bc9b8a0cef1a4c83
    ```
   outputs:
   ```APY for USDC-WFTM is 37.36900950074754```
2. Example run for WFTM-BOO:
    ```shell
    python apy.py 0x21be370d5312f44cb42ce377bc9b8a0cef1a4c83 0x841fad6eae12c286d1fd18d1d525dffa75c7effe
    ```
   outputs:
   ```APY for WFTM-BOO is 2.4817559467878385```