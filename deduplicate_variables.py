import json
import pickle

from disassembly_types import LinearExpr, Variable

with open("generated_data/variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)

with open("generated_data/used_variables.json") as f:
    is_var_used: list[bool] = json.load(f)

last_num_unique = -1
while True:
    values = [
        var.value for i, var in enumerate(variables) if is_var_used[i] and var.value is not None
    ]
    num_unique = len(set(values))
    print(
        f"{num_unique} unique values "
        f"({100 * num_unique / len(values):.2f}% of {len(values)} used variables with values)"
    )
    if num_unique == last_num_unique:
        break
    last_num_unique = num_unique

    remap: dict[LinearExpr, int] = {}
    for i, var in enumerate(variables):
        if is_var_used[i] and var.value is not None and var.value not in remap:
            remap[var.value] = i

    remap_var_index = {
        i: i if var.value is None else remap[var.value]
        for i, var in enumerate(variables)
        if is_var_used[i]
    }

    for i, var in enumerate(variables):
        if is_var_used[i] and var.value is not None:
            var.value = LinearExpr(
                constant=var.value.constant,
                terms=frozenset((remap_var_index[i], coef) for i, coef in var.value.terms),
            )

OUTPUT_FILE = "generated_data/remap_equivalent_variables.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(remap_var_index, f, indent=2)
print(f"Wrote {OUTPUT_FILE} ({len(remap_var_index)} entries)")


# Now, remove unused and duplicate variables and remap the indices.
kept_variable_indices = set(remap_var_index.values())
kept_variables = [var for i, var in enumerate(variables) if i in kept_variable_indices]
remap_index = {
    old: new
    for new, old in enumerate(old for old in range(len(variables)) if old in kept_variable_indices)
}

for var in kept_variables:
    if var.value is not None:
        var.value = LinearExpr(
            constant=var.value.constant,
            terms=frozenset((remap_index[i], coef) for i, coef in var.value.terms),
        )

OUTPUT_FILE = "generated_data/reduced_variables.pickle"
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(kept_variables, f)
print(
    f"Wrote {OUTPUT_FILE} ({len(kept_variables)} variables, "
    f"{100 * len(kept_variables) / len(variables):.2f}% of {len(variables)} original variables)"
)
