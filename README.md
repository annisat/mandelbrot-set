# mandelbrot-set

Beauty of Math
![mandelbrot basic](https://github.com/annisat/mandelbrot-set/blob/main/mandelbrot_ice_series/-0.5%2B0j_1_flame.png)

## to-do
- [x] 4e-8/pixel does not converge properly with 100k iteration. Give it more! => It turns out that the problem is the insufficient number of digits with numpy.csingle.
- [ ] Try numba.cuda for self-made acceleration and compare it with cupy. Reason for this is that mandelbrot set calculation has less to do with matrix calculation than massive complex iteration.
