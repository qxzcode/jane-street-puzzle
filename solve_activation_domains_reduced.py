import shutil
import string
from typing import NamedTuple

from ortools.sat.python import cp_model

import disassembly

state_dict = load_file("model_3_11.safetensors")
state_dict.keys()


def matmul(in_vec, matrix, bias_vec=None) -> list[cp_model.LinearExprT]:
    if bias_vec is None:
        bias_vec = [0] * len(matrix)
    return [
        sum(m * x for m, x in zip(row, in_vec, strict=True) if m != 0) + bias
        for row, bias in zip(matrix, bias_vec, strict=True)
    ]


def get_layer(i: int):
    class Layer(NamedTuple):
        weight: list[list[int]]
        bias: list[int]

    if i < 0:
        i = 5442 + i
    return Layer(
        state_dict[f"{i}.weight"].int().tolist(),
        state_dict[f"{i}.bias"].int().tolist(),
    )


chars = set(string.printable)
chars.remove("\t")
chars.remove("\n")
chars.remove("\x0b")
chars.remove("\x0c")
chars.remove("\r")
input_domain = sorted(ord(c) for c in chars)
print(input_domain)
input_min = min(input_domain)
input_max = max(input_domain)
print(f"range = [{input_min}, {input_max}]")
assert input_domain == list(range(input_min, input_max + 1))

# DEBUG:
# input_min, input_max = 0, 255


# from max_out_value_map import max_out_value_map  # noqa: E402

max_out_value_map = {}  # noqa: F811

var_max_out_value_map = {}  # maps var index -> max value

model = cp_model.CpModel()
in_vec = [
    model.new_int_var(input_min, input_max, f"in_vec[{i}]")
    for i in range(len(get_layer(0).weight[0]))
]
out_vec = in_vec

for target_layer_index in range(0, 5442, 2):
    print("Building model...")

    MAX_OUT_VALUE = +10_000

    def relu(
        m: cp_model.CpModel, in_scalar: cp_model.LinearExprT, max_out_value: int = MAX_OUT_VALUE
    ) -> cp_model.LinearExprT:
        # always create the variable, even if we don't use it, so that var indices are consistent :/
        out_scalar = m.new_int_var(0, max_out_value, "out_scalar")

        assert max_out_value >= 0
        if max_out_value == 0:
            return 0

        # print(f"=================== {type(in_scalar)!r}")
        if type(in_scalar) is int:
            return max(0, in_scalar)
        if type(in_scalar) is cp_model.IntVar:
            lb, ub = in_scalar.proto.domain
            if lb == ub:
                return max(0, lb)
            elif lb >= 0:
                return in_scalar
        # if type(in_scalar) is cp_model._Sum:
        #     var_coef_map, constant = in_scalar.get_integer_var_value_map()
        #     ...

        m.add_max_equality(out_scalar, [0, in_scalar])
        return out_scalar

        # else:
        #     print()
        #     print(repr(in_scalar))
        #     raise ValueError(f"unknown in_scalar type in relu: {type(in_scalar)!r}")

    assert target_layer_index % 2 == 0

    last_model = model.clone()
    last_out_vec = out_vec

    def update_out_vec(model, out_vec):
        layer = get_layer(target_layer_index)
        out_vec = matmul(out_vec, layer.weight, layer.bias)
        out_vec = [
            relu(model, x, max_out_value_map.get((target_layer_index, i), MAX_OUT_VALUE))
            for i, x in enumerate(out_vec)
        ]
        return out_vec

    out_vec = update_out_vec(model, out_vec)
    if all(isinstance(x, int) for x in out_vec) or target_layer_index == "never":
        print(f"{target_layer_index = }")
        print(out_vec)
        # print(var_max_out_value_map)
        exit()
    print(f"{target_layer_index = }")
    print(out_vec)

    num_activations = len(get_layer(target_layer_index).weight)
    for target_activation_index in range(num_activations):
        target = out_vec[target_activation_index]

        key = (target_layer_index, target_activation_index)
        print(f"{target_layer_index = },  {target_activation_index = } ({num_activations} total)")

        if key in max_out_value_map:
            print(f"max_out_value_map[{key}] = {max_out_value_map[key]}  (computed previously)")
        else:
            if type(target) is int:
                print("Target is an int constant; nothing to solve")
                solved_max_value = target
            elif type(target) is cp_model.IntVar and target.index in var_max_out_value_map:
                print("Target is an int var that has already been solved")
                solved_max_value = var_max_out_value_map[target.index]
            else:
                print("Solving...")
                model.maximize(target)

                solver = cp_model.CpSolver()
                solver.parameters.log_search_progress = True
                solver.parameters.max_presolve_iterations = 3
                solver.parameters.max_memory_in_mb = 3_000

                status = solver.solve(model)
                print(f"{solver.wall_time = }")
                print(f"{(status == cp_model.OPTIMAL) = }")

                # get the input vector values
                if status == cp_model.OPTIMAL:
                    solved_max_value = solver.value(target)
                else:
                    print(f"{(status == cp_model.FEASIBLE) = }")
                    print(f"{(status == cp_model.INFEASIBLE) = }")
                    raise Exception("infeasible")

            max_out_value_map[key] = solved_max_value
            print(f"max_out_value_map[{key}] = {max_out_value_map[key]}")

        if type(target) is cp_model.IntVar:
            var_max_out_value_map[target.index] = max_out_value_map[key]

    # print(f"{max_out_value_map}\n")
    with open("max_out_value_map-tmp.py", "w") as f:
        f.write("# AUTO-GENERATED BY solve_activation_domains.py\n")
        f.write(f"max_out_value_map = {max_out_value_map!r}\n")
    shutil.move("max_out_value_map-tmp.py", "max_out_value_map.py")

    model = last_model
    out_vec = last_out_vec
    out_vec = update_out_vec(model, out_vec)

    # break
