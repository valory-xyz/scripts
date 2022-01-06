Description
===========
The `summative_stats.py` script accepts three positional arguments, 
the start date, the pool ids, and the interval to use.
It also accepts one optional argument, the end date.
It outputs the pools' statistics, including the APY.
> 📝 You may run `python summative_stats.py --help` for usage information.

Run
===
This demo uses [poetry](https://python-poetry.org/) for python packaging and dependency management.

Example
-------

Example run:
```shell
python summative_stats.py 1633442859 0x2b4c76d0dc16be1c31d4c1dc53bf9b45987fc75c 0xe120ffbda0d14f3bb6d6053e90e63c572a66a428 86400
```