# TrajecSimu

6-dof trajectory simulation for high-power rockets.  
current version: 3.0 (11/1/2018)

## Description
Solves a 6-dof equation of motion for a trajectory of a transonic high-power rocket.  
Limited to ones without attitude/trajectory control.

Might have some problems on Windows/Linux.

## Usage

### Requirement
Basic python modules: numpy, scipy, pandas, matplotlib are required.  
An external module is used for quaternion computaion, so please install: https://github.com/moble/quaternion


### Install

```sh
$ git clone https://github.com/yamamotsu/TrajecSimu
```

### Demo
`driver_sample.py` is a sample driver code to run simulation. See comments in the file.  

A sample rocket configuration file `sample_config.csv` and a thrust curve `sample_thrust.csv` are available.

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[shugok](https://github.com/shugok)
