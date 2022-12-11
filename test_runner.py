import numpy as np
from minimizers import DividingMethod, GoldenSection, GradientMethod
from methods import broken_curve_method


def test_func_add(x):
    return (x[0] + 5)**2 + x[1]**2


gs_method = GoldenSection()
ds_method = DividingMethod()
gr_method = GradientMethod()


#test_func = lambda x: x**2
test_func = lambda x : np.cos(x)

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


# broken curve
a=0
b=10
L = 2 

x = np.linspace(a, b)
y = np.sin(x)

#xlist, plist = broken_curve_method(math.sin, a, b, L, dividing_method, 100)

#plt.scatter(xlist, plist)
#plt.show()
# end
