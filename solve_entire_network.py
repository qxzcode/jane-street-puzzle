import pickle
import time
from argparse import ArgumentParser
from collections import deque

from ortools.sat.python import cp_model
from tqdm import tqdm

from disassembly_types import Variable

parser = ArgumentParser()
parser.add_argument("--full", action="store_true", help="Solve the full network in one shot")
args = parser.parse_args()


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


minimize_hint_values = [None for _ in variables]
maximize_hint_values = minimize_hint_values.copy()


def build_model(
    target_var_index: int, max_bfs_expansions: int
) -> tuple[cp_model.CpModel, dict[int, cp_model.IntVar]]:
    """Builds a CP-SAT model for the given variable index."""
    assert max_bfs_expansions > 0, "max_bfs_expansions must be greater than 0"

    # print("Building model...")
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
    progress = tqdm(desc="Building model", total=max_bfs_expansions, unit="expansion", leave=False)
    queue = deque([target_var_index])
    visited = set()
    while queue and len(visited) < max_bfs_expansions:
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
        for neighbor_index in neighbors[var_index]:
            if neighbor_index not in visited:
                queue.append(neighbor_index)

        progress.update(1)

    progress.close()
    # print(f"{len(var_index_to_cp_var)} vars in model")

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

    return model, var_index_to_cp_var


def solve_for_min_max(target_var_index: int, max_bfs_expansions: int) -> tuple[int, int]:
    model, var_index_to_cp_var = build_model(target_var_index, max_bfs_expansions)

    # print("Solving...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    # solver.parameters.max_presolve_iterations = 10
    solver.parameters.max_memory_in_mb = 4_096

    def get_optimum_value(maximize: bool) -> int:
        global minimize_hint_values, maximize_hint_values

        # Add hints.
        hint_values = maximize_hint_values if maximize else minimize_hint_values
        model.clear_hints()
        for i, cp_var in var_index_to_cp_var.items():
            if hint_values[i] is not None:
                model.add_hint(cp_var, hint_values[i])

        # Solve.
        target_var = var_index_to_cp_var[target_var_index]
        if maximize:
            model.maximize(target_var)
        else:
            model.minimize(target_var)
        status = solver.solve(model)
        # print(f"{solver.wall_time = }")

        if status != cp_model.OPTIMAL:
            print(f"{(status == cp_model.OPTIMAL) = }")
            print(f"{(status == cp_model.FEASIBLE) = }")
            print(f"{(status == cp_model.INFEASIBLE) = }")
            print(f"{(status == cp_model.UNKNOWN) = }")
            raise Exception("Not optimal")

        # Update hints for the next solve.
        for i, cp_var in var_index_to_cp_var.items():
            hint_values[i] = solver.value(cp_var)
        if not maximize:
            maximize_hint_values = hint_values.copy()

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


if args.full:
    # Solve the entire network in one shot.
    model, var_index_to_cp_var = build_model(0, len(variables))
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_presolve_iterations = 10
    solver.parameters.max_memory_in_mb = 4_096

    status = solver.solve(model)
    save_variable_values(solver, var_index_to_cp_var)
    if status != cp_model.OPTIMAL:
        print(f"{(status == cp_model.OPTIMAL) = }")
        print(f"{(status == cp_model.FEASIBLE) = }")
        print(f"{(status == cp_model.INFEASIBLE) = }")
        print(f"{(status == cp_model.UNKNOWN) = }")
        raise Exception("Not optimal")

    exit(0)


TOTAL_HOURS_TARGET = 12
TOTAL_SECS_TARGET = 60 * 60 * TOTAL_HOURS_TARGET
target_end_time = time.perf_counter() + TOTAL_SECS_TARGET

avg_time_by_mbe = {}
EMA_LAMBDA = 0.9

with open("generated_data/bounds.txt", "w") as bounds_file:
    prog = tqdm(reversed(range(len(variables))), desc="Solving", unit="var", total=len(variables))
    for i in prog:
        now = time.perf_counter()
        target_duration_this_var = (target_end_time - now) / (i + 1)
        target_end_time_this_var = now + target_duration_this_var

        # CP-SAT can't handle the last 2 layers, so skip them.
        if variables[i].name.startswith(("x5438_", "x5440_")):
            continue

        if variables[i].min_value == variables[i].max_value:
            # print(
            #     f"Variable {variables[i].name} is a constant ({variables[i].min_value}); skipping."
            # )
            min_value, max_value = (variables[i].min_value, variables[i].max_value)
        else:
            # Determine the max_bfs_expansions based on the time remaining.
            max_bfs_expansions = 2
            while True:
                if max_bfs_expansions * 2 not in avg_time_by_mbe or (
                    avg_time_by_mbe[max_bfs_expansions * 2] > target_duration_this_var
                ):
                    break
                max_bfs_expansions *= 2

            min_value, max_value = 0, None
            while min_value != max_value:
                # print(
                #     "\n\n"
                #     f"Solving bounds for {variables[i].name} with {max_bfs_expansions=}"
                #     f" (time remaining for this var: {max(0, target_end_time_this_var - time.perf_counter()):.3f}s)"
                #     "\n\n"
                # )

                solve_start = time.perf_counter()

                min_value, max_value = solve_for_min_max(i, max_bfs_expansions)
                variables[i].min_value = min_value
                variables[i].max_value = max_value

                solve_duration = time.perf_counter() - solve_start

                if max_bfs_expansions in avg_time_by_mbe:
                    avg_time_by_mbe[max_bfs_expansions] = (
                        EMA_LAMBDA * avg_time_by_mbe[max_bfs_expansions]
                        + (1 - EMA_LAMBDA) * solve_duration
                    )
                else:
                    avg_time_by_mbe[max_bfs_expansions] = solve_duration

                time_left_after_next_solve = (
                    target_end_time_this_var
                    - time.perf_counter()
                    - avg_time_by_mbe.get(max_bfs_expansions * 2, 0.0)
                )

                if min_value == max_value:
                    prog.write(
                        f"\n\nSucceeded for {variables[i].name} with {max_bfs_expansions=}\n\n"
                    )
                    break
                elif time_left_after_next_solve < 0:
                    # prog.write(
                    #     f"\n\nDoubling max_bfs_expansions would end up {-time_left_after_next_solve:.3f}s over budget; stopping\n\n"
                    # )
                    break
                else:
                    max_bfs_expansions *= 2

        bounds_file.write(f"{variables[i].name:<9} {min_value:>3} {max_value:>3}\n")
        bounds_file.flush()


print(f"{avg_time_by_mbe=}")
print("\nAll done.")
