import json
import pickle

from disassembly_types import Variable

with open("generated_data/variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)


is_var_visited = [False] * len(variables)


def visit_var(var_index: int):
    if not is_var_visited[var_index]:
        is_var_visited[var_index] = True
        var = variables[var_index]
        if var.value is not None:
            for term_var_index, _ in var.value.terms:
                visit_var(term_var_index)


# Visit all variables that are (transitively) used in the value of the last (output) variable.
visit_var(len(variables) - 1)

print(
    f"Visited {sum(is_var_visited)} out of {len(variables)} variables ({100 * sum(is_var_visited) / len(variables):.2f}%)"
)


OUTPUT_FILE = "generated_data/used_variables.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(is_var_visited, f, indent=2)
print(f"Wrote {OUTPUT_FILE} ({len(is_var_visited)} entries)")
