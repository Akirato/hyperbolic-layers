import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from ..manifolds.base import Manifold

class RAdam(keras.optimizers.Optimizer):
    """Optimizer that implements the Riemannian Adam algorithm."""

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        stabilize=None,
        manifold = Manifold(),
        name="RiemannianAdam",
        **kwargs,
    ):
        """Construct a new Riemannian Adam optimizer.
        Becigneul, Gary, and Octavian-Eugen Ganea. "Riemannian Adaptive
        Optimization Methods." International Conference on Learning
        Representations. 2018.
        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.001.
          beta_1: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use. The exponential decay
            rate for the 1st moment estimates. Defaults to 0.9.
          beta_2: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use, The exponential decay
            rate for the 2nd moment estimates. Defaults to 0.999.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
          amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond". Defaults to `False`.
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          name: Optional name for the operations created when applying gradients.
            Defaults to "RiemannianAdam".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """

        super(RAdam, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        self.stabilize = stabilize
        self.manifold = manifold

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(RAdam, self)._prepare_local(
            var_device, var_dtype, apply_state
        )

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]["lr_t"] * (
            math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(RiemannianAdam, self).set_weights(weights)

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        grad = self.manifold.egrad2rgrad(var, grad)

        alpha = (
            coefficients["lr_t"]
            * math_ops.sqrt(1 - coefficients["beta_2_power"])
            / (1 - coefficients["beta_1_power"])
        )
        m.assign_add((grad - m) * (1 - coefficients["beta_1_t"]))
        v.assign_add(
            (self.manifold.inner(var, grad, grad, keepdims=True) - v)
            * (1 - coefficients["beta_2_t"])
        )

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat.assign(math_ops.maximum(vhat, v))
            v = vhat
        var_t = self.manifold.expmap(
            var, -(m * alpha) / (math_ops.sqrt(v) + coefficients["epsilon"])
        )
        m.assign(self.manifold.parallel_transport(var, var_t, m))
        var.assign(var_t)

        if self.stabilize is not None:
            self._stabilize(var)

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        grad = self.manifold.egrad2rgrad(var, grad)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t_values = (
            array_ops.gather(m, indices) * coefficients["beta_1_t"]
            + m_scaled_g_values
        )

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (
            self.manifold.inner(var, grad, grad, keepdims=True)
            * coefficients["one_minus_beta_2_t"]
        )
        v_t_values = (
            array_ops.gather(v, indices) * coefficients["beta_2_t"]
            + v_scaled_g_values
        )

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat.scatter_max(ops.IndexedSlices(v_t_values, indices))
            v_t_values = array_ops.gather(vhat, indices)

        var_values = array_ops.gather(var, indices)
        var_t_values = self.manifold.expmap(
            var_values,
            -(m_t_values * coefficients["lr"])
            / (math_ops.sqrt(v_t_values) + coefficients["epsilon"]),
        )
        m_t_transp = self.manifold.parallel_transport(var_values, var_t_values, m_t_values)

        m.scatter_update(ops.IndexedSlices(m_t_transp, indices))
        v.scatter_update(ops.IndexedSlices(v_t_values, indices))
        var.scatter_update(ops.IndexedSlices(var_t_values, indices))

        if self.stabilize is not None:
            self._stabilize(var)

    @def_function.function(experimental_compile=True)
    def _stabilize(self, var):
        if math_ops.floor_mod(self.iterations, self.stabilize) == 0:
            m = self.get_slot(var, "m")
            var.assign(self.manifold.proj(var))
            m.assign(self.manifold.proju(var, m))

    def get_config(self):
        config = super(RiemannianAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "stabilize": self.stabilize,
            }
        )
        return config