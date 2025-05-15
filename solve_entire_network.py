import pickle

from ortools.sat.python import cp_model
from tqdm import tqdm

from disassembly_types import Variable
from max_out_value_map import max_out_value_map

# Load the variables.
with open("variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)

# Tighten the max_values using the max_out_value_map.
for i, var in enumerate(variables):
    if not var.is_max_value_tight:
        var.max_value = max_out_value_map[i, var.name]


def solve_for_min_max(target_var_index: int) -> tuple[int, int] | None:
    print("Building model...")
    model = cp_model.CpModel()

    var_index_to_cp_var: dict[int, cp_model.IntVar] = {}
    progress = tqdm(desc="Building model", total=len(variables), unit="var")

    def get_var_layer_index(var_index: int) -> int:
        name = variables[var_index].name
        return 0 if name.startswith("in") else int(name.split("_")[0][1:])

    def get_cp_var(var_index: int, min_constraint_layer_index: int = 0) -> cp_model.IntVar:
        var = variables[var_index]
        layer_index = get_var_layer_index(var_index)

        if var_index not in var_index_to_cp_var:
            var_index_to_cp_var[var_index] = model.new_int_var(
                var.min_value, var.max_value, var.name
            )
            progress.update(1)

            if var.value is not None and layer_index >= min_constraint_layer_index:
                linear_expr = (
                    sum(
                        coef * get_cp_var(var_index, min_constraint_layer_index)
                        for var_index, coef in var.value.terms
                    )
                    + var.value.constant
                )

                # Don't actually add the constraint for the last 2 layers, because CP-SAT can't handle those raw.
                if not var.name.startswith(("x5438_", "x5440_")):
                    model.add_max_equality(var_index_to_cp_var[var_index], [0, linear_expr])

        return var_index_to_cp_var[var_index]

    # Load all the variables and intermediate constraints.
    # for i in range(len(variables)):
    #     get_cp_var(i)
    get_cp_var(
        len(variables) - 1,
        min_constraint_layer_index=get_var_layer_index(target_var_index) - 50,
    )
    progress.close()
    print(f"{len(var_index_to_cp_var)} vars in model")

    if target_var_index not in var_index_to_cp_var:
        return None

    # Add the output constraints.
    variables_by_name = {var.name: (i, var) for i, var in enumerate(variables)}
    for i in range(16, 32):
        var_index, var = variables_by_name[f"x5438_{i}"]
        linear_expr = var.value
        assert linear_expr is not None, f"var {var.name} has no value"

        # assert all(variables[i].max_value <= 1 for i, _ in linear_expr.terms)

        # target = -linear_expr.constant
        # assert 0 <= target <= 255

        linear_expr_value = linear_expr.constant + sum(
            coef * var_index_to_cp_var[var_index] for var_index, coef in linear_expr.terms
        )
        model.add(linear_expr_value == 0)

    print("Solving...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    # solver.parameters.max_presolve_iterations = 10
    solver.parameters.max_memory_in_mb = 3_000

    def get_optimum_value(maximize: bool) -> int:
        target_var = var_index_to_cp_var[target_var_index]
        if maximize:
            model.maximize(target_var)
        else:
            model.minimize(target_var)
        status = solver.solve(model)
        print(f"{solver.wall_time = }")

        print(f"{(status == cp_model.OPTIMAL) = }")
        print(f"{(status == cp_model.FEASIBLE) = }")
        print(f"{(status == cp_model.INFEASIBLE) = }")
        print(f"{(status == cp_model.UNKNOWN) = }")

        if status != cp_model.OPTIMAL:
            raise Exception("infeasible")

        return solver.value(target_var)

    # Get the min and max values.
    min_value = get_optimum_value(maximize=False)
    max_value = get_optimum_value(maximize=True)
    return min_value, max_value


def save_variable_values(
    solver: cp_model.CpSolver, var_index_to_cp_var: dict[int, cp_model.IntVar]
):
    """Saves the values of the variables in the solution."""
    with open("variable_values.json", "w") as f:
        f.write("{\n")
        for i, var in enumerate(variables):
            if i > 0:
                f.write(",\n")
            value = solver.value(var_index_to_cp_var[i]) if i in var_index_to_cp_var else "null"
            f.write(f'  "{var.name}": {value}')
        f.write("\n}\n")


with open("generated_data/bounds.txt", "w") as bounds_file:
    for i in tqdm(
        reversed(range(len(variables))), desc="Solving", unit="var", total=len(variables)
    ):
        # CP-SAT can't handle the last 2 layers, so skip them.
        if variables[i].name.startswith(("x5438_", "x5440_")):
            continue

        if (result := solve_for_min_max(i)) is not None:
            min_value, max_value = result
            bounds_file.write(f"{variables[i].name:<10} {min_value:>3} {max_value:>3}\n")
            variables[i].min_value = min_value
            variables[i].max_value = max_value


print("\nAll done.")
