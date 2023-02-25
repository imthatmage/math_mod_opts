import numpy as np
import math
from gradient_method import GradientMethod
from golden_section import GoldenSection
from dividing_method import DividingMethod
from gradient_projection import GradientProjectionMethod
from broken_lines_method import BrokenLinesMethod
from gradient_conditional import GradientConditionalMethod


def test_func_add(x):
    #return (x[0] + 5)**2 + math.sin(10*(x[0] - np.pi/4)) + x[1]**2
    return (x[0] + 5)**2 + x[1]**2

def test_func_add1(x):
    return (x[0] + 5)**2 + math.sin(10*(x[0] - np.pi/4)) + x[1]**2 + (x[2] - 1)**2
    #return (x[0] + 5)**2 + x[1]**2


gs_method = GoldenSection()
ds_method = DividingMethod()
gr_method = GradientMethod()
gp_method = GradientProjectionMethod(area_name='sphere', R=4, C=[0, 0])
bl_method = BrokenLinesMethod()

#test_func = lambda x: x**2
#test_func = lambda x : np.cos(x)
test_func = lambda x: x**2#   + np.sin(10*(x - np.pi/4))

print(f"Dividing Method {ds_method.minimize(test_func, -5, 6)}")
print()
# print(f"Golden section: {gs_method.minimize(test_func, -5, 6)}")
# print(f"BrokenLinesMethod: {bl_method.minimize(test_func, -5, 5, 100, 2, 100)}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='armicho')}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='monotone')}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='apriori')}")
# 
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='armicho')}")
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='monotone')}")
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='apriori')}")

print()

# for name in ['monotone', 'armicho', 'apriori']:
#     # gr_method = GradientMethod(method_name=name, n=1000, verbose=False, full_vis=False)
#     # print(f"Gradient method: {name}")
#     # print(f"Gradient method: {gr_method.minimize(test_func, 1, [3])}")
#     # gr_method = GradientMethod(method_name=name, n=1000, verbose=True, full_vis=True)
#     # print(f"Gradient method: {gr_method.minimize(test_func_add, 2, [2, 5])}")
#     # print()
#     gr_method = GradientMethod(method_name=name, n=1000, verbose=True, full_vis=True)
#     print(f"Gradient method: {gr_method.minimize(test_func_add1, 3, [2, 5, 3])}")
#     print()

# print()

# for name in ['monotone', 'armicho', 'apriori']:
#     gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=2, C=0)
#     print(f"GMP: {name}")
#     print(f"GMP: {gp_method.minimize(test_func, 1, [3])}")
#     gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=2, C=[-5, 0], n=100, verbose=True, full_vis=True)
#     print(f"GMP: {gp_method.minimize(test_func_add, 2, [2, 5])}")
#     print()

# for name in ['monotone', 'armicho', 'apriori']:
#     gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=2, C=0)
#     print(f"GMP: {name}")
#     print(f"GMP: {gp_method.minimize(test_func, 1, [3])}")
#     gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=10, C=[-5, 2, 3], n=100, verbose=True, full_vis=True)
#     print(f"GMP: {gp_method.minimize(test_func_add1, 3, [2, 5, 3])}")
#     print()



for name in ['monotone', 'armicho', 'apriori']:
    gp_method = GradientConditionalMethod(method_name=name, area_name='sphere', R=2, C=0)
    print(f"GMC: {name}")
    print(f"GMC: {gp_method.minimize(test_func, 1, [3])}")
    gp_method = GradientConditionalMethod(method_name=name, area_name='sphere', R=4, C=[0, 0], n=1000, verbose=True, full_vis=True)
    print(f"GMC: {gp_method.minimize(test_func_add, 2, [4, 12])}")
    print()
