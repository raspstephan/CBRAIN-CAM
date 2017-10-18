# Stephan's project notebook

### October 18

Trying to get main.py to run. Problems
- no variable LAT in mean and std files --> Create myself in data_exploration.ipynb

```
<Closed HDF5 file>
Exception in thread Thread-7:
Traceback (most recent call last):
  File "/home/s/S.Rasp/repositories/CBRAIN/dataLoad.py", line 155, in accessData
    arr = fileReader[k][:,s:s+l].T
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper (/home/ilan/minonda/conda-bld/h5py_1496889914775/work/h5py/_objects.c:2846)
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper (/home/ilan/minonda/conda-bld/h5py_1496889914775/work/h5py/_objects.c:2804)
  File "/home/s/S.Rasp/.conda/envs/cbrain/lib/python3.6/site-packages/h5py/_hl/group.py", line 169, in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper (/home/ilan/minonda/conda-bld/h5py_1496889914775/work/h5py/_objects.c:2846)
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper (/home/ilan/minonda/conda-bld/h5py_1496889914775/work/h5py/_objects.c:2804)
  File "h5py/h5o.pyx", line 190, in h5py.h5o.open (/home/ilan/minonda/conda-bld/h5py_1496889914775/work/h5py/h5o.c:3740)
ValueError: Not a location (Invalid object id)
```
Happens at the end of the epochs, training works probably still. Doesn't appear for convolution with 50 epochs

- Tensorboard visualisation doesn't work 
