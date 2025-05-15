from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Variable:
    name: str
    min_value: int
    max_value: int
    is_max_value_tight: bool
    value: "LinearExpr | None"
    use_count: int = 0

    @property
    def layer_index(self) -> int | None:
        if self.name.startswith("in"):
            return None
        return int(self.name[1 : self.name.index("_")])

    def format(self, variables: "list[Variable]") -> str:
        if self.value:
            string = f"{self.name} = relu({self.value.format(variables)})  # [{self.min_value}, {self.max_value}]"
        else:
            string = f"{self.name} = ...  # [{self.min_value}, {self.max_value}]"

        if self.is_max_value_tight:
            string += "  <- tight upper bound"

        if self.use_count == 0:
            string += "; unused"
        else:
            string += f"; use_count={self.use_count}"

        return string


class LinearExpr(NamedTuple):
    constant: int
    terms: frozenset[tuple[int, int]]

    def min_value(self, variables: list[Variable]) -> int:
        return self.constant + sum(
            min(coef * variables[var].min_value, coef * variables[var].max_value)
            for var, coef in self.terms
        )

    def max_value(self, variables: list[Variable]) -> int:
        return self.constant + sum(
            max(coef * variables[var].min_value, coef * variables[var].max_value)
            for var, coef in self.terms
        )

    def __add__(self, rhs: "LinearExpr | int") -> "LinearExpr":
        if isinstance(rhs, int):
            return self._replace(constant=self.constant + rhs)
        else:
            assert isinstance(rhs, LinearExpr)
            terms = dict(self.terms)
            for var, coef in rhs.terms:
                if var in terms:
                    terms[var] += coef
                    if terms[var] == 0:
                        del terms[var]
                else:
                    terms[var] = coef
            assert all(coef != 0 for coef in terms.values())

            return LinearExpr(
                constant=self.constant + rhs.constant,
                terms=frozenset(terms.items()),
            )

    def __radd__(self, lhs: "LinearExpr | int") -> "LinearExpr":
        return self + lhs

    def __mul__(self, rhs: int) -> "LinearExpr":
        assert isinstance(rhs, int)
        if rhs == 1:
            return self
        return LinearExpr(
            constant=self.constant * rhs,
            terms=frozenset((var, coef * rhs) for var, coef in ([] if rhs == 0 else self.terms)),
        )

    def __rmul__(self, lhs: int) -> "LinearExpr":
        return self * lhs

    def format(self, variables: list[Variable]) -> str:
        string = ""

        for i, (var, coef) in enumerate(sorted(self.terms)):
            assert coef != 0
            var_name = variables[var].name
            if coef == 1:
                string += var_name if i == 0 else f" + {var_name}"
            elif coef == -1:
                string += f"-{var_name}" if i == 0 else f" - {var_name}"
            else:
                if i == 0:
                    string += f"{coef}*{var_name}"
                elif coef > 0:
                    string += f" + {coef}*{var_name}"
                else:
                    string += f" - {-coef}*{var_name}"

        if self.constant != 0:
            if string == "":
                string += str(self.constant)
            elif self.constant > 0:
                string += f" + {self.constant}"
            else:
                string += f" - {-self.constant}"

        return string

    @staticmethod
    def from_var(var: int) -> "LinearExpr":
        return LinearExpr(constant=0, terms=frozenset([(var, 1)]))

    @staticmethod
    def from_const(constant: int) -> "LinearExpr":
        return LinearExpr(constant=constant, terms=frozenset())
