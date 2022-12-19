# Here you can see some guides how to use this package of programs


## Sympy

### How to write your expression to add it as argument

1. Follow standard documentation of sympy (https://docs.sympy.org/latest/modules/functions/index.html).
2. Use the English alphabet in order to write arguments.


### Here some examples of function providing

**-f (--func_expr)**

- "sin(a)cos(b)**2 + (tan(c)*exp(-d*5))**e + sinh(f)"
- "5**a + b + c:

## Keys

**-m (--method_name)**

Available names: gradient_descent (short variant: gd), dividing_segment (ds), golden_section (gs), gradient_projection(gp)

**-gu (--gradient_updater)**

Should be provided if you use gradient_descent (default value: armicho).
Available updaters: 'monotone', 'armicho', 'apriori'.

**-p (--init_point)**

Initial point (should be provided if you use gradient_descent)

example for number_of_arguments=3: -l 0 1 3 


**for any additional information you  can always write python handler.py --help**


## Examples

1. python handler.py -f "cos(a)" -n 1 -m gd -mi 1500 -p -5 -pp True -gu armicho
2. python handler.py -f "sin(a) + cos(b)" -n 2 -m gd -mi 1500 -p -5 2 -pp True -gu monotone
3. python handler.py -f "cos(a)" -n 1 -m ds -mi 1500 -a -2 -b 5
4. python handler.py -f "(a+6)**2" -n 1 -m gs -mi 1500 -a -5 -b
5. python handler.py -f "(a+5)**2 + b**2 + (c-3)**2" -n 3 -m gp -mi 50 -p 2 5 -4 -pp True -gu armicho -aname sphere -R 10 -C 0 0 0 -v True -fv True
6. python handler.py -f "(a+5)**2 + sin(10*(b - pi/4))" -n 2 -m gp -mi 50 -p 2 5 -pp True -gu monotone -aname sphere -R 2 -C -5 0  -v True -fv True

