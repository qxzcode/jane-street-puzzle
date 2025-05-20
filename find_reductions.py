import math
import pickle
import sys

from ortools.sat.python import cp_model
from pydantic import BaseModel, TypeAdapter

from disassembly_types import Variable

# with open("generated_data/variables.pickle", "rb") as f:
with open("generated_data/reduced_variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)
variables_by_name = {var.name: (i, var) for i, var in enumerate(variables)}
len(variables)

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


# Find variables where all its inputs are constants.
def is_constant(var: Variable) -> bool:
    return var.min_value == var.max_value


def var_str(var: Variable) -> str:
    return f"{var.name}[{var.min_value},{var.max_value}]"


for var in variables:
    if is_constant(var) or var.value is None:
        continue
    if all(is_constant(variables[input_index]) for input_index, _ in var.value.terms):
        print(
            f"Variable {var_str(var)} is constant (inputs: {[var_str(variables[input_index]) for input_index, _ in var.value.terms]})"
        )

# Build the (undirected) neighbor graph.
neighbors: list[list[int]] = [[] for _ in variables]
for i, var in enumerate(variables):
    if var.value is not None:
        for var_index, _ in var.value.terms:
            neighbors[i].append(var_index)
            neighbors[var_index].append(i)

### Find sub-networks that have a large number of variables which are completely determined by a small input domain. ###


model = cp_model.CpModel()

is_input = [model.new_bool_var(f"is_input({var.name})") for var in variables]
is_determined = [model.new_bool_var(f"is_determined({var.name})") for var in variables]
is_eliminated = [model.new_bool_var(f"is_eliminated({var.name})") for var in variables]

# Constraint: the size of the input domain must be at most 100.
LOG_SCALE_FACTOR = 10
model.add(
    sum(
        ii * math.ceil(math.log2(var.max_value - var.min_value + 1) * LOG_SCALE_FACTOR)
        for ii, var in zip(is_input, variables)
        if not is_constant(var)
    )
    <= math.ceil(math.log2(100) * LOG_SCALE_FACTOR)
)

for var, var_neighbors, is_input_var, is_determined_var, is_eliminated_var in zip(
    variables, neighbors, is_input, is_determined, is_eliminated
):
    # Constraint: If a variable is input, it is determined.
    model.add_implication(is_input_var, is_determined_var)

    # Constraint: constants are determined, eliminated, and not input.
    if is_constant(var):
        model.add(is_input_var == 0)
        model.add(is_determined_var == 1)
        model.add(is_eliminated_var == 1)
    else:
        # Constraint: A variable is eliminated iff it and all of its neighbors are determined.
        literals = [is_determined[i] for i in var_neighbors] + [is_determined_var]
        model.add_bool_and(literals).only_enforce_if(is_eliminated_var)
        model.add_bool_or(~lit for lit in literals).only_enforce_if(~is_eliminated_var)

        if var.value is None:
            model.add(is_determined_var == is_input_var)
        else:
            assert len(var.value.terms) > 0, f"Variable {var.name} has no terms."

            # Constraints: A variable is determined iff all of its inputs are determined.
            model.add_bool_and(
                is_determined[input_index] for input_index, _ in var.value.terms
            ).only_enforce_if(is_determined_var, ~is_input_var)
            model.add_bool_or(
                ~is_determined[input_index] for input_index, _ in var.value.terms
            ).only_enforce_if(~is_determined_var)

# # Maximize the number of determined (non-const) variables that are not input.
# model.maximize(
#     sum(
#         is_determined_var - is_input_var
#         for var, is_input_var, is_determined_var in zip(variables, is_input, is_determined)
#         if not is_constant(var)
#     )
# )

# Maximize the number of eliminated (non-const) variables.
model.maximize(
    sum(
        is_eliminated_var
        for var, is_eliminated_var in zip(variables, is_eliminated)
        if not is_constant(var)
    )
)


class Reduction(BaseModel):
    inputs: list[str]
    determined: list[str]
    eliminated: list[str]


def find_reduction(max_time_in_seconds: float) -> Reduction:
    def print_solution(
        solver: cp_model.CpSolverSolutionCallback | cp_model.CpSolver, indent: int = 0
    ):
        sys.stdout.flush()

        indent = " " * indent
        print(f"{indent}Input (non-const) variables:")
        for var, ii, ie in zip(variables, is_input, is_eliminated):
            if not is_constant(var) and solver.value(ii):
                e_str = " (eliminated)" if solver.value(ie) else ""
                print(f"{indent}    {var.name:<9} in [{var.min_value}, {var.max_value}]{e_str}")

        num_det = sum(
            not is_constant(var)
            and solver.value(is_determined_var)
            and not solver.value(is_input_var)
            for var, is_input_var, is_determined_var in zip(variables, is_input, is_determined)
        )
        print(f"{indent}Determined (non-const) variables: {num_det}")
        for var, is_input_var, is_determined_var, is_eliminated_var in zip(
            variables, is_input, is_determined, is_eliminated
        ):
            if (
                not is_constant(var)
                and solver.value(is_determined_var)
                and not solver.value(is_input_var)
            ):
                e_str = " (eliminated)" if solver.value(is_eliminated_var) else ""
                print(f"{indent}    {var.name:<9} in [{var.min_value}, {var.max_value}]{e_str}")

        print(f"{indent}^ Solution (objective={int(solver.objective_value)})")

        sys.stdout.flush()

    class MySolutionCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            print_solution(self, indent=4)

    # Temporary constraints:
    # for i, var in enumerate(variables):
    #     if var.name.startswith("in") or var.max_value > 2:
    #         model.add(is_input[i] == 0)

    ### Solve ###
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    solver.parameters.max_memory_in_mb = 2 * 1_000
    solver.parameters.symmetry_level = 3
    solver.parameters.max_presolve_iterations = 1
    status = solver.solve(model, MySolutionCallback())

    print(f"{status == cp_model.OPTIMAL=}")
    print(f"{status == cp_model.FEASIBLE=}")
    print(f"{status == cp_model.INFEASIBLE=}")
    print(f"{status == cp_model.UNKNOWN=}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print()
        print_solution(solver)
        return Reduction(
            inputs=[
                var.name
                for var, is_input_var in zip(variables, is_input)
                if solver.value(is_input_var)
            ],
            determined=[
                var.name
                for var, is_determined_var in zip(variables, is_determined)
                if solver.value(is_determined_var) and not is_constant(var)
            ],
            eliminated=[
                var.name
                for var, is_eliminated_var in zip(variables, is_eliminated)
                if solver.value(is_eliminated_var) and not is_constant(var)
            ],
        )
    else:
        raise Exception(f"Solver failed with status {status}")


if __name__ == "__main__":
    reductions = []
    eliminated_vars = set()
    while True:
        reduction = find_reduction(max_time_in_seconds=60)
        reductions.append(reduction)

        assert all(var not in eliminated_vars for var in reduction.eliminated)
        eliminated_vars.update(reduction.eliminated)
        print(
            f"Total eliminated variables: {len(eliminated_vars)} / {len(variables)} ({len(eliminated_vars) / len(variables):.2%})"
        )

        # Add constraints: not allowed to eliminate variables that were eliminated in a previous iteration.
        for var_name in reduction.eliminated:
            var_index, var = variables_by_name[var_name]
            model.add(is_eliminated[var_index] == 0)

        with open("generated_data/reductions.json", "wb") as f:
            f.write(TypeAdapter(list[Reduction]).dump_json(reductions, indent=2))
