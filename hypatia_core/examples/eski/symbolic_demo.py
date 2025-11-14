# examples/grad_demo.py
from hypatia_core import Symbol

x = Symbol.variable("x")
c5 = Symbol.const(5.0)

# f(x) = 2x
f = x * Symbol.const(2.0)

# loss = (f - 5)^2  ->  (f - 5) * (f - 5)
loss = (f - c5) * (f - c5)

dl_dx = loss.derivative("x").simplify()
print("loss:", loss)
print("dl/dx:", dl_dx)   # beklenen: 2*(2x - 5) * 2 = 4x - 20 (simplify düzeyine göre eşdeğer form)
