import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import grad
from jax import jacobian
from jax import jacfwd, jacrev
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from time import time


def norm(x):
    x = x - x.mean(0)
    return x / x.std(0)
    

def jax_basics():
    # Create a JAX array
    x = jnp.array([1.0, 2.0, 3.0])
    print("JAX Array:", x)

    # Perform basic operations
    y = x * 2
    print("After multiplication:", y)

    # Use JAX's built-in functions
    z = jnp.sin(x)
    print("Sine of x:", z)

    # Attempt to mutate the JAX array (this will raise an error)
    try:
        x[0] = 10.0
    except TypeError as e:
        print("Error while trying to mutate JAX array:", e)
    
    # Successful update using x.at[0].set(10.0)
    x = x.at[0].set(10.0)
    print("After successful update:", x)

    # check x device
    print("Device of x", x.devices())
    print("Sharding of x", x.sharding)

    # testing JIT
    norm_compiled = jit(norm)
    np.random.seed(1701)
    x = jnp.array(np.random.rand(10000, 10))
    np.allclose(norm(x), norm_compiled(x), atol=1E-6)

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def first_finite_difference(f, x, eps=1e-3):
    # central difference approximation of the gradient
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in 
                      jnp.eye(len(x))])

def hessian(f):
    return jacfwd(jacrev(f))

def jax_grad():
    # Define a simple function
    def f(x):
        return jnp.sum(x ** 2)

    # Compute the gradient using JAX
    grad_f = grad(f)

    # Evaluate the gradient at a point
    x = jnp.array([1.0, 2.0, 3.0])
    gradient = grad_f(x)
    print("Gradient of f at x:", gradient)

    x_small = jnp.arange(3.)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))

    print(first_finite_difference(sum_logistic, x_small))

    # Compose JIT and grad
    print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

    # Get the jacobian of a function
    jacobian_fn = jacobian(jnp.exp)
    jacobian_matrix = jacobian_fn(x_small)
    print("Jacobian of sum_logistic at x_small:", jacobian_matrix)

    # Hessian
    print("Hessian of sum_logistic at x_small:", hessian(sum_logistic)(x_small))

def apply_matrix(mat, x):
    return jnp.dot(mat, x)

@jit
def batched_apply_matrix(mat, batched_x):
    return jnp.dot(batched_x, mat.T)

@jit
def vmap_batched_apply_matrix(mat, batched_x):
    return vmap(apply_matrix, in_axes=(None, 0))(mat, batched_x)

def jax_vec():
    key = random.key(1701)
    key1, key2 = random.split(key)
    mat = random.normal(key1, (1000, 1000))
    batched_x = random.normal(key2, (10, 1000))
    start = time()
    result1 = batched_apply_matrix(mat, batched_x)
    end = time()
    print(f"Time for batched matrix multiplication: {end - start}")
    start = time()
    result2 = vmap_batched_apply_matrix(mat, batched_x)
    end = time()
    print(f"Time for vmap batched matrix multiplication: {end - start}")

def jax_random():
    key = random.key(43)
    print(key)
    print(random.normal(key))
    print(random.normal(key))
    for i in range(3):
        new_key, subkey = random.split(key)
        del key  # The old key is consumed by split() -- we must never use it again.

        val = random.normal(subkey)
        del subkey  # The subkey is consumed by normal().

        print(f"draw {i}: {val}")
        key = new_key  # new_key is safe to use in the next iteration.

@jit
def f(x):
    y = jnp.sin(x)
    print("print(x) ->", x)
    print("print(y) ->", y)
    jax.debug.print("print(x) ->{x}", x=x)
    jax.debug.print("print(y) ->{y}", y=y)


if __name__ == "__main__":
    jax_basics()
    jax_grad()
    jax_vec()
    jax_random()
    result = f(2)