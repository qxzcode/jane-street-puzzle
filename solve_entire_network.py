import pickle
from collections import deque

from ortools.sat.python import cp_model
from tqdm import tqdm

from disassembly_types import Variable

# Load the variables.
with open("generated_data/reduced_variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)
variables_by_name = {var.name: (i, var) for i, var in enumerate(variables)}


# Tighten the bounds using bounds.txt.
print("Applying existing bounds...")
with open("generated_data/bounds.txt") as bounds_file:
    for line in bounds_file:
        name, min_value, max_value = line.split()
        min_value, max_value = int(min_value), int(max_value)
        if name in variables_by_name:
            var_index, var = variables_by_name[name]
            assert var.min_value <= min_value <= max_value <= var.max_value, (
                f"Invalid bounds for {name}: {var.min_value} <= {min_value} <= {max_value} <= {var.max_value}"
            )
            var.min_value = min_value
            var.max_value = max_value
        else:
            print(f"Ignoring unknown variable {name} ∈ [{min_value}, {max_value}]")


# Build the (undirected) neighbor graph.
neighbors = [[] for _ in variables]
for i, var in enumerate(variables):
    if var.value is not None:
        for var_index, _ in var.value.terms:
            neighbors[i].append(var_index)
            neighbors[var_index].append(i)


def solve_for_min_max(target_var_index: int) -> tuple[int, int] | None:
    print("Building model...")
    model = cp_model.CpModel()

    var_index_to_cp_var: dict[int, cp_model.IntVar] = {}

    def get_cp_var(var_index: int) -> cp_model.IntVar:
        if var_index not in var_index_to_cp_var:
            var = variables[var_index]
            var_index_to_cp_var[var_index] = model.new_int_var(
                var.min_value, var.max_value, var.name
            )

        return var_index_to_cp_var[var_index]

    # Perform a BFS to build a limited-size graph of variables and constraints around the target variable.
    MAX_EXPANSIONS = 100
    progress = tqdm(desc="Building model", total=MAX_EXPANSIONS, unit="expansion")
    queue = deque([target_var_index])
    visited = set()
    while queue and len(visited) < MAX_EXPANSIONS:
        var_index = queue.popleft()
        if var_index in visited:
            continue
        visited.add(var_index)

        # Add the constraint (except for the last 2 layers, because CP-SAT can't handle those raw).
        var = variables[var_index]
        if not (var.value is None or var.name.startswith(("x5438_", "x5440_"))):
            linear_expr = (
                sum(coef * get_cp_var(var_index) for var_index, coef in var.value.terms)
                + var.value.constant
            )
            model.add_max_equality(get_cp_var(var_index), [0, linear_expr])

        # Add neighbors to the queue.
        print(f"Variable {var.name} has {len(neighbors[var_index])} neighbors")
        for neighbor_index in neighbors[var_index]:
            if neighbor_index not in visited:
                queue.append(neighbor_index)

        progress.update(1)

    progress.close()
    print(f"{len(var_index_to_cp_var)} vars in model")

    assert target_var_index in var_index_to_cp_var

    # Add the output constraints.
    for i in range(16, 32):
        var_index, var = variables_by_name[f"x5438_{i}"]
        linear_expr = var.value
        assert linear_expr is not None, f"var {var.name} has no value"

        # assert all(variables[i].max_value <= 1 for i, _ in linear_expr.terms)

        # target = -linear_expr.constant
        # assert 0 <= target <= 255

        linear_expr_value = linear_expr.constant + sum(
            coef * get_cp_var(var_index) for var_index, coef in linear_expr.terms
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

        if variables[i].min_value == variables[i].max_value:
            print(
                f"Variable {variables[i].name} is a constant ({variables[i].min_value}); skipping."
            )
            result = (variables[i].min_value, variables[i].max_value)
        else:
            result = solve_for_min_max(i)

        if result is not None:
            min_value, max_value = result

            bounds_file.write(f"{variables[i].name:<9} {min_value:>3} {max_value:>3}\n")
            bounds_file.flush()

            variables[i].min_value = min_value
            variables[i].max_value = max_value


print("\nAll done.")
