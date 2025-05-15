import importlib
import shutil
import string
from dataclasses import dataclass
from typing import NamedTuple

from safetensors.numpy import load_file
from tqdm import tqdm

from disassembly_types import LinearExpr, Variable

state_dict = load_file("model_3_11.safetensors")


variables: list[Variable] = []


def new_var(
    name: str, min_value: int, max_value: int, is_max_value_tight: bool, value: "LinearExpr | None"
) -> int:
    # Increment use counts
    if value is not None:
        for var, _ in value.terms:
            variables[var].use_count += 1

    variables.append(Variable(name, min_value, max_value, is_max_value_tight, value))
    return len(variables) - 1


def matmul(
    in_vec: list[LinearExpr], matrix: list[list[int]], bias_vec: list[int] | None = None
) -> list[LinearExpr]:
    if bias_vec is None:
        bias_vec = [0] * len(matrix)
    return [
        sum(m * x for m, x in zip(row, in_vec, strict=True) if m != 0) + LinearExpr.from_const(bias)
        for row, bias in zip(matrix, bias_vec, strict=True)
    ]


def relu(
    in_scalar: LinearExpr, var_name_suffix: str, max_out_value: int | None = None
) -> LinearExpr:
    if in_scalar.max_value(variables) <= 0 or max_out_value == 0:
        return LinearExpr.from_const(0)

    if in_scalar.min_value(variables) >= 0:
        return in_scalar

    is_max_value_tight = max_out_value is not None
    if max_out_value is None:
        max_out_value = in_scalar.max_value(variables)
    assert max_out_value > 0

    var_name = f"b{var_name_suffix}" if max_out_value == 1 else f"x{var_name_suffix}"
    out_scalar = LinearExpr.from_var(
        new_var(var_name, 0, max_out_value, is_max_value_tight, in_scalar)
    )
    return out_scalar


def get_layer(i: int):
    class Layer(NamedTuple):
        weight: list[list[int]]
        bias: list[int]

    if i < 0:
        i = 5442 + i
    return Layer(
        state_dict[f"{i}.weight"].astype(int).tolist(),
        state_dict[f"{i}.bias"].astype(int).tolist(),
    )


chars = set(string.printable)
chars.remove("\t")
chars.remove("\n")
chars.remove("\x0b")
chars.remove("\x0c")
chars.remove("\r")
input_domain = sorted(ord(c) for c in chars)
input_min = min(input_domain)
input_max = max(input_domain)
# print(f"input range = [{input_min}, {input_max}]")
assert input_domain == list(range(input_min, input_max + 1))

# DEBUG:
# input_min, input_max = 0, 255


max_out_value_map: dict[tuple[int, int], int] = importlib.import_module(
    "max_out_value_map"
).max_out_value_map
# max_out_value_map = {}  # noqa: F811


in_vec = [
    LinearExpr.from_var(new_var(f"in{i}", input_min, input_max, True, None))
    for i in range(len(get_layer(0).weight[0]))
]
out_vec = in_vec

for target_layer_index in tqdm(range(0, 5442, 2)):
    assert target_layer_index % 2 == 0

    layer = get_layer(target_layer_index)
    out_vec = matmul(out_vec, layer.weight, layer.bias)
    out_vec = [
        relu(x, f"{target_layer_index}_{i}", max_out_value_map.get((target_layer_index, i)))
        for i, x in enumerate(out_vec)
    ]

    # if target_layer_index == 200:
    #     break


# Compute variable lifetimes
alive_variables_by_index = [None] * len(variables)
alive_variables = {len(variables) - 1}
for i, var in reversed(list(enumerate(variables))):
    alive_variables_by_index[i] = alive_variables.copy()

    try:
        alive_variables.remove(i)
    except KeyError:
        assert var.use_count == 0

    if var.value:
        for var, _ in var.value.terms:
            alive_variables.add(var)
alive_variables_by_index: list[set[int]]


# Print stats
values = [v.value for v in variables if v.value is not None]
unique_values = set(values)
print(f"#  Total variables: {len(values)}")
print(f"# Unique variables: {len(unique_values)}")
print()


if __name__ == "__main__":
    # Print the variables as pseudocode
    last_prefix = "in"
    for var_index, var in enumerate(variables):
        prefix = "in" if var.name.startswith("in") else var.name[1 : var.name.index("_")]
        if prefix != last_prefix:
            print()
            alive_vars = alive_variables_by_index[var_index - 1]
            print(f"# Alive:  {', '.join(variables[i].name for i in sorted(alive_vars))}")
            alive_layers = {
                (-1 if variables[i].layer_index is None else variables[i].layer_index)
                for i in sorted(alive_vars)
            }
            print(
                f"#     from layers:  {', '.join('<input>' if i == -1 else str(i) for i in sorted(alive_layers))}"
            )
            print()
            last_prefix = prefix
        print(var.format(variables))
