"""
Lambda Parser for Telepathic Benchmark v2.

Parses Python lambda syntax into an AST representation,
then converts to De Bruijn indices for BLC compilation.

Checkpoint 1.1: Lambda parser (Python lambda syntax → AST)
Checkpoint 1.2: De Bruijn index conversion
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Union


# =============================================================================
# AST Representation (Named Variables)
# =============================================================================

@dataclass
class Var:
    """Variable reference."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class Abs:
    """Lambda abstraction: λname.body"""
    param: str
    body: LambdaTerm

    def __repr__(self) -> str:
        return f"(λ{self.param}.{self.body})"


@dataclass
class App:
    """Function application: func(arg)"""
    func: LambdaTerm
    arg: LambdaTerm

    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


# Union type for any lambda term
LambdaTerm = Union[Var, Abs, App]


# =============================================================================
# De Bruijn Index Representation
# =============================================================================

@dataclass
class DBVar:
    """De Bruijn variable: index counts binding depth."""
    index: int

    def __repr__(self) -> str:
        return str(self.index)


@dataclass
class DBAbs:
    """De Bruijn abstraction: λ.body (no parameter name needed)."""
    body: DBTerm

    def __repr__(self) -> str:
        return f"(λ.{self.body})"


@dataclass
class DBApp:
    """De Bruijn application: func arg."""
    func: DBTerm
    arg: DBTerm

    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


# Union type for De Bruijn terms
DBTerm = Union[DBVar, DBAbs, DBApp]


# =============================================================================
# Parser: Python Lambda String → Named AST
# =============================================================================

class ParseError(Exception):
    """Raised when parsing fails."""
    pass


def parse(source: str) -> LambdaTerm:
    """
    Parse a Python lambda expression into our AST.

    Args:
        source: Python lambda expression as string.
                Can include variable assignments for named definitions.

    Returns:
        LambdaTerm: Our AST representation.

    Examples:
        >>> parse("lambda x: x")
        (λx.x)
        >>> parse("lambda x: lambda y: x")
        (λx.(λy.x))
        >>> parse("lambda f: lambda x: f(x)")
        (λf.(λx.(f x)))
    """
    try:
        tree = ast.parse(source, mode='eval')
    except SyntaxError as e:
        raise ParseError(f"Invalid Python syntax: {e}")

    return _convert_node(tree.body, {})


def parse_with_definitions(source: str) -> LambdaTerm:
    """
    Parse a multi-line program with variable definitions.

    The last expression is returned with all definitions inlined.

    Args:
        source: Multi-line Python with assignments and a final expression.

    Returns:
        LambdaTerm: Fully inlined AST.

    Example:
        >>> code = '''
        ... ZERO = lambda f: lambda x: x
        ... SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
        ... SUCC(ZERO)
        ... '''
        >>> parse_with_definitions(code)
        # Returns SUCC(ZERO) with SUCC and ZERO inlined
    """
    try:
        tree = ast.parse(source, mode='exec')
    except SyntaxError as e:
        raise ParseError(f"Invalid Python syntax: {e}")

    # Build environment from assignments
    env: dict[str, LambdaTerm] = {}
    result: LambdaTerm | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Variable assignment: NAME = expr
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ParseError("Only simple assignments (NAME = expr) are allowed")
            name = node.targets[0].id
            env[name] = _convert_node(node.value, env)
        elif isinstance(node, ast.Expr):
            # Expression statement: the final result
            result = _convert_node(node.value, env)
        else:
            raise ParseError(f"Unsupported statement type: {type(node).__name__}")

    if result is None:
        raise ParseError("No expression found in source")

    return result


def _convert_node(node: ast.expr, env: dict[str, LambdaTerm]) -> LambdaTerm:
    """Convert a Python AST node to our lambda AST."""

    if isinstance(node, ast.Lambda):
        # Lambda abstraction: lambda x: body
        if len(node.args.args) != 1:
            raise ParseError("Lambda must have exactly one argument")
        if node.args.defaults or node.args.kw_defaults:
            raise ParseError("Default arguments not allowed")
        if node.args.vararg or node.args.kwarg:
            raise ParseError("*args and **kwargs not allowed")

        param = node.args.args[0].arg
        # Add parameter to a new scope (shadowing)
        new_env = dict(env)
        new_env.pop(param, None)  # Remove any outer binding
        body = _convert_node(node.body, new_env)
        return Abs(param, body)

    elif isinstance(node, ast.Name):
        # Variable reference
        name = node.id
        if name in env:
            # Inline the definition
            return env[name]
        return Var(name)

    elif isinstance(node, ast.Call):
        # Function application: f(x)
        if node.keywords:
            raise ParseError("Keyword arguments not allowed")
        if len(node.args) != 1:
            raise ParseError("Function application must have exactly one argument")

        func = _convert_node(node.func, env)
        arg = _convert_node(node.args[0], env)
        return App(func, arg)

    elif isinstance(node, ast.BinOp):
        raise ParseError(f"Binary operators not allowed in pure lambda calculus: {ast.dump(node)}")

    elif isinstance(node, ast.IfExp):
        raise ParseError("Conditional expressions not allowed in pure lambda calculus")

    elif isinstance(node, ast.Constant):
        raise ParseError(f"Literals not allowed in pure lambda calculus: {node.value}")

    else:
        raise ParseError(f"Unsupported AST node type: {type(node).__name__}")


# =============================================================================
# De Bruijn Conversion: Named AST → De Bruijn AST
# =============================================================================

class ConversionError(Exception):
    """Raised when De Bruijn conversion fails."""
    pass


def to_debruijn(term: LambdaTerm) -> DBTerm:
    """
    Convert a named lambda term to De Bruijn indices.

    De Bruijn indices count how many binders are between the variable
    and its binding lambda. Index 0 = bound by innermost lambda.

    Args:
        term: Lambda term with named variables.

    Returns:
        DBTerm: Term with De Bruijn indices instead of names.

    Raises:
        ConversionError: If a free variable is encountered.

    Examples:
        >>> to_debruijn(parse("lambda x: x"))
        (λ.0)
        >>> to_debruijn(parse("lambda x: lambda y: x"))
        (λ.(λ.1))
        >>> to_debruijn(parse("lambda x: lambda y: y"))
        (λ.(λ.0))
    """
    return _to_debruijn_with_env(term, [])


def _to_debruijn_with_env(term: LambdaTerm, env: list[str]) -> DBTerm:
    """
    Convert with an environment tracking bound variables.

    env[0] is the most recently bound variable (innermost).
    """
    if isinstance(term, Var):
        # Look up the variable in the environment
        try:
            index = env.index(term.name)
            return DBVar(index)
        except ValueError:
            raise ConversionError(f"Free variable not allowed: {term.name}")

    elif isinstance(term, Abs):
        # Add parameter to front of environment (innermost binding)
        new_env = [term.param] + env
        body = _to_debruijn_with_env(term.body, new_env)
        return DBAbs(body)

    elif isinstance(term, App):
        func = _to_debruijn_with_env(term.func, env)
        arg = _to_debruijn_with_env(term.arg, env)
        return DBApp(func, arg)

    else:
        raise ConversionError(f"Unknown term type: {type(term)}")


# =============================================================================
# Utility Functions
# =============================================================================

def parse_to_debruijn(source: str) -> DBTerm:
    """
    Convenience function: parse and convert to De Bruijn in one step.

    Args:
        source: Python lambda expression.

    Returns:
        DBTerm: De Bruijn indexed term.
    """
    return to_debruijn(parse(source))


def parse_program_to_debruijn(source: str) -> DBTerm:
    """
    Convenience function: parse multi-line program and convert to De Bruijn.

    Args:
        source: Multi-line Python with definitions.

    Returns:
        DBTerm: Fully inlined De Bruijn indexed term.
    """
    return to_debruijn(parse_with_definitions(source))
