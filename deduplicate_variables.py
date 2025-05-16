import json
import pickle

from disassembly_types import LinearExpr, Variable

with open("generated_data/variables.pickle", "rb") as f:
    variables: list[Variable] = pickle.load(f)

last_num_unique = -1
while True:
    num_unique = len(set(var.value for var in variables if var.value is not None))
    print(
        f"{num_unique} unique values ({100 * num_unique / len(variables):.2f}% of {len(variables)})"
    )
    if num_unique == last_num_unique:
        break
    last_num_unique = num_unique

    remap: dict[LinearExpr, int] = {}
    for i, var in enumerate(variables):
        if var.value is not None and var.value not in remap:
            remap[var.value] = i

    remap_var_index = [
        i if var.value is None else remap[var.value] for i, var in enumerate(variables)
    ]

    for var in variables:
        if var.value is not None:
            var.value = LinearExpr(
                constant=var.value.constant,
                terms=frozenset((remap_var_index[i], coef) for i, coef in var.value.terms),
            )

OUTPUT_FILE = "generated_data/remap_equivalent_variables.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(remap_var_index, f, indent=2)
print(f"Wrote {OUTPUT_FILE} ({len(remap_var_index)} entries)")
