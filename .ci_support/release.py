import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def get_exact_pin_versions(pyproject_content):
    pyproject_dict = tomllib.loads(pyproject_content)
    versions = {}
    for section, key in [("build-system", "requires"), ("project", "dependencies")]:
        for dep in pyproject_dict.get(section, {}).get(key, []):
            if "==" in dep:
                name, version = dep.split("==", 1)
                versions[name] = version
    return versions


def extract_lower_constraint(constraint):
    if not constraint:
        return None
    for part in constraint.split(","):
        normalized = part.strip().replace(" ", "")
        if not normalized:
            continue
        if normalized.startswith("=="):
            return f">={normalized[2:]}"
        if normalized.startswith(">="):
            return f">={normalized[2:]}"
        if normalized.startswith("="):
            return f">={normalized[1:]}"
        if normalized.startswith(">"):
            return normalized
    return None


def sanitize_version(version):
    match = re.match(r"^([0-9A-Za-z.+!_-]+)", version)
    if match:
        return match.group(1)
    return None


def parse_conda_dependency(dep):
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", dep.strip())
    if not match:
        return None, None
    name, constraint = match.groups()
    constraint_token = constraint.strip().split()[0] if constraint.strip() else ""
    lower_constraint = extract_lower_constraint(constraint_token)
    if not lower_constraint:
        return name, None
    operator = ">=" if lower_constraint.startswith(">=") else ">"
    version = lower_constraint[2:] if operator == ">=" else lower_constraint[1:]
    sanitized_version = sanitize_version(version=version)
    if not sanitized_version:
        return name, None
    return name, f"{operator}{sanitized_version}"


def get_bracket_delta(line):
    in_single_quote = False
    in_double_quote = False
    delta = 0
    previous = ""
    for char in line:
        if char == "'" and not in_double_quote and previous != "\\":
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote and previous != "\\":
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            if char == "[":
                delta += 1
            elif char == "]":
                delta -= 1
        previous = char
    return delta


def get_env_versions(env_content):
    versions = {}
    in_dependencies = False
    for raw_line in env_content.splitlines():
        line = raw_line.strip()
        if line == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if raw_line and not raw_line.startswith((" ", "\t", "-")):
            break
        if not line.startswith("-"):
            continue
        dep = line.lstrip("-").strip()
        if dep == "pip:":
            msg = (
                "Nested pip dependencies are not supported in environment.yml for "
                "release lower-bound extraction."
            )
            raise ValueError(msg)
        name, version = parse_conda_dependency(dep=dep)
        if name and version:
            versions[name] = version
    return versions


def to_release_constraint(dep, high_version, low_constraint):
    if low_constraint == f">={high_version}":
        return f"{dep}=={high_version}"
    return f"{dep}{low_constraint},<={high_version}"


def update_dependencies(pyproject_content, version_low_dict, version_high_dict):
    missing_dependencies = sorted(
        [dep for dep in version_high_dict if dep not in version_low_dict]
    )
    if missing_dependencies:
        msg = "Missing lower-bound versions for dependencies: " + ", ".join(
            missing_dependencies
        )
        raise ValueError(msg)

    replacement_dict = {}
    for dep, high_version in version_high_dict.items():
        old = f"{dep}=={high_version}"
        replacement_dict[old] = to_release_constraint(
            dep, high_version, version_low_dict[dep]
        )

    updated_lines = []
    current_section = None
    in_target_list = False
    bracket_balance = 0
    for line in pyproject_content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.strip("[]")
            in_target_list = False
            bracket_balance = 0
        if current_section in ["build-system", "project"] and (
            "requires = [" in line or "dependencies = [" in line
        ):
            in_target_list = True
            bracket_balance = get_bracket_delta(line=line)
        if in_target_list:
            for old, new in replacement_dict.items():
                line = line.replace(f'"{old}"', f'"{new}"').replace(
                    f"'{old}'", f"'{new}'"
                )
            if "requires = [" not in line and "dependencies = [" not in line:
                bracket_balance += get_bracket_delta(line=line)
            if bracket_balance <= 0:
                in_target_list = False
                bracket_balance = 0
        updated_lines.append(line)
    return "".join(updated_lines)


def update_pyproject_for_release(pyproject_content, env_content):
    version_high_dict = get_exact_pin_versions(pyproject_content=pyproject_content)
    version_low_dict = get_env_versions(env_content=env_content)
    return update_dependencies(
        pyproject_content=pyproject_content,
        version_low_dict=version_low_dict,
        version_high_dict=version_high_dict,
    )


if __name__ == "__main__":
    with open("pyproject.toml", "r", encoding="utf-8") as file:
        setup_content = file.read()

    with open("environment.yml", "r", encoding="utf-8") as file:
        env_content = file.read()

    setup_content_new = update_pyproject_for_release(
        pyproject_content=setup_content, env_content=env_content
    )

    with open("pyproject.toml", "w", encoding="utf-8") as file:
        file.write(setup_content_new)
