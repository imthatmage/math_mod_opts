import numpy as np
from gradient_method import GradientMethod
from golden_section import GoldenSection
from dividing_method import DividingMethod
from gradient_projection import GradientProjectionMethod
from methods import broken_curve_method


def test_func_add(x):
    return (x[0] + 5)**2 + x[1]**2


gs_method = GoldenSection()
ds_method = DividingMethod()
gr_method = GradientMethod()
gp_method = GradientProjectionMethod(area_name='sphere', R=4, C=[0, 0])

#test_func = lambda x: x**2
#test_func = lambda x : np.cos(x)
test_func = lambda x: x**2 + np.sin(10*(x - np.pi/4))

print(f"Dividing Method {ds_method.minimize(test_func, -5, 6)}")
print()
print(f"Golden section: {gs_method.minimize(test_func, -5, 6)}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='armicho')}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='monotone')}")
# print(f"Gradient method: {gradient_method(test_func, 1, [3], method_name='apriori')}")
# 
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='armicho')}")
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='monotone')}")
# print(f"Gradient method: {gradient_method(test_func_add, 2, [2, 5], method_name='apriori')}")

print()

for name in ['monotone', 'armicho', 'apriori']:
    gr_method = GradientMethod(method_name=name, n=1000)
    print(f"Gradient method: {name}")
    print(f"Gradient method: {gr_method.minimize(test_func, 1, [3])}")
    print(f"Gradient method: {gr_method.minimize(test_func_add, 2, [2, 5])}")
    print()

print()

for name in ['monotone', 'armicho', 'apriori']:
    gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=4, C=0)
    print(f"GMP: {name}")
    print(f"GMP: {gp_method.minimize(test_func, 1, [3])}")
    gp_method = GradientProjectionMethod(method_name=name, area_name='sphere', R=4, C=[0, 0], n=20, verbose=True)
    print(f"GMP: {gp_method.minimize(test_func_add, 2, [2, 5])}")
    print()